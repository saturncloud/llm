from __future__ import annotations

from queue import Queue
from multiprocessing import Pipe, Process, set_start_method
from typing import Any, Dict, Iterable, List, Optional, Union

import torch

from llm.inference.base import InferenceEngine
from llm.inference.transformer import TransformersEngine
from llm.model_configs import ModelConfig


class MultiprocessEngine(InferenceEngine):
    """
    Run inference engines in background processes with queued requests/responses

    Enables thread-safe non-blocking inference across multiple devices
    """

    def __init__(self, workers: List[WorkerPipe]):
        self.workers = workers
        self.closed = False
        self.queue: Queue[WorkerPipe] = Queue(len(self.workers))
        for worker in self.workers:
            self.queue.put(worker)

    @classmethod
    def from_model_config(
        cls,
        model_config: ModelConfig,
        num_workers: Optional[int] = None,
        wait: bool = True,
        load_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> MultiprocessEngine:
        # Required for CUDA
        set_start_method("spawn")

        if num_workers is None:
            # TODO: Should we also check CUDA_VISIBLE_DEVICES?
            num_workers = torch.cuda.device_count()

        workers: List[WorkerPipe] = []
        for i in range(num_workers):
            worker = WorkerPipe()
            watch_proc = Process(
                None,
                cls._transformers_worker,
                args=[worker, model_config, i],
                kwargs={"signal_ready": wait, "load_kwargs": load_kwargs, **kwargs},
                daemon=True,
            )
            watch_proc.start()
            workers.append(worker)

        if wait:
            for worker in workers:
                # Collect initial sentinel values indicating the workers have loaded
                worker.get_response()

        return cls(workers)

    @staticmethod
    def _transformers_worker(
        pipe: WorkerPipe,
        model_config: ModelConfig,
        local_rank: int,
        signal_ready: bool = False,
        load_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        load_kwargs = load_kwargs or {}
        load_kwargs.setdefault("device_map", {"": local_rank})
        engine = TransformersEngine.from_model_config(
            model_config, load_kwargs=load_kwargs, **kwargs
        )
        if signal_ready:
            pipe.send_response(None)
        while True:
            request = pipe.get_request()
            for text in engine.generate_stream(request.prompt, **request.kwargs):
                pipe.send_response(text)

            # None indicates to the requester that the stream is completed
            pipe.send_response(None)

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        echo_prompt: bool = False,
        stop_token_ids: Optional[List[int]] = None,
        stop_strings: Union[str, List[str]] = "",
        **kwargs,
    ) -> Iterable[str]:
        request = StreamRequest(
            prompt,
            max_new_tokens=max_new_tokens,
            echo_prompt=echo_prompt,
            stop_token_ids=stop_token_ids,
            stop_strings=stop_strings,
            **kwargs,
        )
        # Wait for a worker to be available
        worker = self.queue.get()
        try:
            # Clear any leftover responses from a previous query
            worker.clear_responses()
            worker.send_request(request)
            while True:
                # Yield results until sentinel value is recieved
                text = worker.get_response()
                if text is not None:
                    yield text
                else:
                    # Stream completed
                    break

            self.queue.task_done()
        finally:
            if not self.closed:
                # Add pipe back to queue to process new requests
                self.queue.put(worker)

    def close(self):
        self.closed = True
        for pipe in self.workers:
            pipe.close()


class StreamRequest:
    """
    Wraps an inference request to be processed by a worker
    """

    def __init__(self, prompt: str, **kwargs) -> None:
        self.prompt = prompt
        self.kwargs = kwargs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            **self.kwargs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StreamRequest:
        return cls(**data)


class WorkerPipe:
    """
    Manages communication between an inference worker and the main process
    """

    def __init__(self):
        self.parent_conn, self.child_conn = Pipe()

    def close(self):
        self.parent_conn.close()
        self.child_conn.close()

    ### Main proc methods ####
    def send_request(self, request: StreamRequest):
        # Send as dict to make pickling more reliable
        self.parent_conn.send(request.to_dict())

    def get_response(self) -> Optional[str]:
        return self.parent_conn.recv()

    def clear_responses(self):
        while self.parent_conn.poll():
            self.parent_conn.recv_bytes()

    ### Worker methods ####
    def get_request(self) -> StreamRequest:
        return StreamRequest.from_dict(self.child_conn.recv())

    def send_response(self, response: Optional[str]):
        self.child_conn.send(response)
