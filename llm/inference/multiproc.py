from __future__ import annotations

from queue import Queue
from multiprocessing import Pipe, Process, set_start_method
from threading import Lock, Thread
from time import time
from typing import Any, Dict, Iterable, List, Optional, Union, overload
from uuid import uuid4

import torch

from llm.inference.base import InferenceEngine
from llm.inference.transformer import InferenceRequest, InferenceResponse, TransformersEngine
from llm.model_configs import ModelConfig


class MultiprocessEngine(InferenceEngine):
    """
    Run the transformers engines in background processes with queued requests/responses

    Enables thread-safe non-blocking inference across multiple devices
    """

    def __init__(self, workers: List[WorkerPipe], max_pending: int = -1):
        self._workers = workers
        self._closed = False
        self._requests: Queue[InferenceRequest] = Queue(max_pending)
        self._active_streams: Dict[str, Queue[InferenceResponse]] = {}
        self._streams_lock = Lock()

        for worker in self._workers:
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
            try:
                request = self._requests.get()
                # TODO: Scheduling across workers
                self._workers[0].send_request(request)
            except OSError:
                # Workers closed
                break

    def _response_manager(self, worker: WorkerPipe):
        """
        Watches for responses from a worker, and sends
        them to the associated output stream
        """
        while True:
            try:
                response = worker.get_response()
            except EOFError:
                # Worker closed
                break

            if response:
                uid = response.uid
                stream = self._active_streams.get(uid)
                if stream:
                    if response.stopped:
                        self.cancel_request(uid)
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
                request = pipe.get_request(wait)
                if request:
                    requests.append(request)
                    if not wait:
                        start = time()
                delta = time() - start

            for response in engine.generate_batch_stream(requests):
                pipe.send_response(response)

    def add_request(self, request: InferenceRequest) -> Queue[InferenceResponse]:
        """
        Add a request to the queue, and return a response queue
        """
        queue: Queue[InferenceResponse] = Queue(-1)
        with self._streams_lock:
            self._active_streams[request.uid] = queue
        self._requests.put(request)
        return queue

    def cancel_request(self, uid: str):
        with self._streams_lock:
            self._active_streams.pop(uid, None)

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        echo_prompt: bool = False,
        stop_token_ids: Optional[List[int]] = None,
        stop_strings: Union[str, List[str]] = "",
        **kwargs,
    ) -> Iterable[str]:
        request = InferenceRequest(
            prompt,
            max_new_tokens=max_new_tokens,
            echo_prompt=echo_prompt,
            stop_token_ids=stop_token_ids,
            stop_strings=stop_strings,
            **kwargs,
        )
        stream = self.add_request(request)
        try:
            while True:
                response = stream.get()
                yield response.output
                if response.stopped:
                    break
        finally:
            self.cancel_request(request.uid)

    def close(self):
        self._closed = True
        for pipe in self._workers:
            pipe.close()


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
    def send_request(self, request: InferenceRequest):
        # Send as dict to make pickling more reliable
        self.parent_conn.send(request.to_dict())

    def get_response(self, timeout: Optional[float] = None) -> Optional[InferenceResponse]:
        if timeout is not None and not self.parent_conn.poll(timeout):
            return None
        response = self.parent_conn.recv()
        return InferenceResponse.from_dict(response) if response else None

    def clear_responses(self):
        while self.parent_conn.poll():
            self.parent_conn.recv_bytes()

    ### Worker methods ####
    @overload
    def get_request(self) -> InferenceRequest:
        pass

    @overload
    def get_request(self, timeout: None = None) -> InferenceRequest:
        pass

    @overload
    def get_request(self, timeout: Optional[float] = None) -> Optional[InferenceRequest]:
        pass

    def get_request(self, timeout: Optional[float] = None) -> Optional[InferenceRequest]:
        if timeout is not None and not self.child_conn.poll(timeout):
            return None
        return InferenceRequest.from_dict(self.child_conn.recv())

    def send_response(self, response: Optional[InferenceResponse]):
        self.child_conn.send(response.to_dict() if response else None)
