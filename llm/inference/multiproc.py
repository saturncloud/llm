from __future__ import annotations

from queue import Queue
from multiprocessing import Pipe, Process, set_start_method
from threading import Thread
from time import time
from typing import Any, Dict, Iterable, List, Optional, Union, overload
from uuid import uuid4

import torch

from llm.inference.base import InferenceEngine
from llm.inference.transformer import TransformersEngine, InferenceRequest
from llm.model_configs import ModelConfig


class MultiprocessEngine(InferenceEngine):
    """
    Run inference engines in background processes with queued requests/responses

    Enables thread-safe non-blocking inference across multiple devices
    """

    def __init__(self, workers: List[WorkerPipe], max_pending: int = -1):
        self.workers = workers
        self.closed = False
        self.requests: Queue[StreamRequest] = Queue(max_pending)
        self.active_streams: Dict[str, Queue[StreamResponse]] = {}

        for worker in self.workers:
            Thread(None, self._response_manager, args=[worker], daemon=True).start()
        Thread(None, self._requests_manager, daemon=True).start()

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

    def _requests_manager(self):
        """
        Watches for requests, and schedules them to workers
        """
        while True:
            request = self.requests.get()
            # TODO: Scheduling across workers
            self.workers[0].send_request(request)

    def _response_manager(self, worker: WorkerPipe):
        """
        Watches for responses from a worker, and sends
        them to the associated output stream
        """
        while True:
            response = worker.get_response()
            if response:
                uid = response.uid
                stream = self.active_streams.get(uid)
                if stream:
                    if response.stopped:
                        self.active_streams.pop(uid, None)
                    stream.put(response)

    @staticmethod
    def _transformers_worker(
        pipe: WorkerPipe,
        model_config: ModelConfig,
        local_rank: int,
        max_wait: float = 1.0,
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
            start = time()
            delta: float = 0.0

            requests: List[InferenceRequest] = []
            while len(engine.pending) < engine.batch_size and delta < max_wait:
                # Wait indefinitely for the first request, collect additional
                # requests over a short window before processing.
                wait = max_wait - delta if requests else None
                stream_req = pipe.get_request(wait)
                if stream_req:
                    req = InferenceRequest(stream_req.prompt, **stream_req.kwargs)
                    requests.append(req)
                    if not wait:
                        start = time()
                delta = time() - start

            for state in engine.run_batch(requests):
                pipe.send_response(StreamResponse(state.req.uid, state.output, state.stopped, state.stopped_reason))

    def add_request(self, request: StreamRequest) -> Queue[StreamResponse]:
        """
        Add a request to the queue, and return a response queue
        """
        queue: Queue[StreamResponse] = Queue(-1)
        self.active_streams[request.uid] = queue
        self.requests.put(request)
        return queue

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
        stream = self.add_request(request)
        while True:
            response = stream.get()
            yield response.output
            if response.stopped:
                break

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
        self.uid = uuid4().hex
        self.kwargs = kwargs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "uid": self.uid,
            **self.kwargs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StreamRequest:
        return cls(**data)


class StreamResponse:
    def __init__(self, uid: str, output: str, stopped: bool = False, stopped_reason: str = "") -> None:
        self.uid = uid
        self.output = output
        self.stopped = stopped
        self.stopped_reason = stopped_reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "output": self.output,
            "stopped": self.stopped,
            "stopped_reason": self.stopped_reason,
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

    def get_response(self) -> Optional[StreamResponse]:
        response = self.parent_conn.recv()
        if response is None:
            return response
        return StreamResponse.from_dict(response)

    def clear_responses(self):
        while self.parent_conn.poll():
            self.parent_conn.recv_bytes()

    ### Worker methods ####
    @overload
    def get_request(self) -> StreamRequest:
        pass

    @overload
    def get_request(self, timeout: None = None) -> StreamRequest:
        pass

    @overload
    def get_request(self, timeout: Optional[float] = None) -> Optional[StreamRequest]:
        pass

    def get_request(self, timeout: Optional[float] = None) -> Optional[StreamRequest]:
        if timeout is not None and not self.child_conn.poll(timeout):
            return None
        return StreamRequest.from_dict(self.child_conn.recv())

    def send_response(self, response: Optional[StreamResponse]):
        self.child_conn.send(response.to_dict() if response else response)
