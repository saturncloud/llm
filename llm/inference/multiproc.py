from __future__ import annotations

from queue import Empty, Queue
from multiprocessing import Pipe, Process, set_start_method
from threading import Lock, Thread
from time import time
from typing import Any, Dict, Iterable, List, Optional, Union, overload

import torch

from llm.inference.base import InferenceEngine
from llm.inference.transformer import TransformersEngine
from llm.inference.types import InferenceRequest, InferenceResponse
from llm.model_configs import ModelConfig
from llm.utils.logs import get_logger

logger = get_logger()


class MultiprocessEngine(InferenceEngine):
    """
    Run one or more TransformersEngine in background processes with queued requests/responses

    Enables thread-safe non-blocking inference across multiple devices
    """

    def __init__(self, workers: List[WorkerPipe], batch_size: int = 8, max_delay: float = 0.5, max_pending: int = -1):
        self._workers = workers
        self._closed = False
        self.batch_size = batch_size
        self.max_delay = max_delay
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
        batch_size: int = 8,
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
                kwargs={"signal_ready": wait, "load_kwargs": load_kwargs, "batch_size": batch_size},
                daemon=True,
            )
            watch_proc.start()
            workers.append(worker)

        if wait:
            for worker in workers:
                # Collect initial sentinel values indicating the workers have loaded
                worker.get_response()

        return cls(workers, batch_size=batch_size, **kwargs)

    def _requests_manager(self):
        """
        Watches for requests, and schedules them to workers
        """
        next_worker = 0
        while True:
            batch = self._collect_batch()
            try:
                self._workers[next_worker].send_request(batch)
            except OSError:
                # Workers closed
                break
            next_worker = (next_worker + 1) % len(self._workers)

    def _collect_batch(self) -> List[InferenceRequest]:
        requests: List[InferenceRequest] = []
        delta = 0.0
        start = time()
        while len(requests) < self.batch_size and delta < self.max_delay:
            # Wait indefinitely for the first request, collect additional
            # requests over a short window before sending to be processed.
            wait = self.max_delay - delta if requests else None
            try:
                req = self._requests.get(timeout=wait)
            except Empty:
                break

            requests.append(req)
            if not wait:
                start = time()
            delta = time() - start
        return requests

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
            try:
                requests = pipe.get_request()
                if not isinstance(requests, list):
                    requests = [requests]
                for response in engine.generate_batch_stream(requests):
                    pipe.send_response(response)
            except EOFError:
                # Worker closed
                break
            except OSError:
                # Worker closed
                break
            except Exception as e:
                logger.error(e, exc_info=True)
                for response in engine.clear_all("internal error"):
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
        input: str,
        max_new_tokens: int = 256,
        echo: bool = False,
        stop_token_ids: Optional[List[int]] = None,
        stop_strings: Union[str, List[str]] = "",
        timeout: Optional[int] = None,
        **kwargs,
    ) -> Iterable[str]:
        request = InferenceRequest(
            input,
            max_new_tokens=max_new_tokens,
            echo=echo,
            stop_token_ids=stop_token_ids,
            stop_strings=stop_strings,
            **kwargs,
        )
        for resp in self.generate_response_stream(request, timeout=timeout):
            yield resp.output

    def generate_response_stream(self, request: InferenceRequest, timeout: Optional[int] = None) -> Iterable[InferenceResponse]:
        stream = self.add_request(request)
        try:
            start = time()
            while timeout is None or timeout > 0:
                response = stream.get(timeout=timeout)
                yield response

                if response.stopped:
                    break
                if timeout is not None:
                    delta = time() - start
                    timeout -= delta
        except Empty:
            pass
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
    def send_request(self, request: Union[InferenceRequest, List[InferenceRequest]]):
        # Send as dict(s) to make pickling more reliable
        if isinstance(request, list):
            data = [req.to_dict() for req in request]
        else:
            data = request.to_dict()
        self.parent_conn.send(data)

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
    def get_request(self) -> Union[InferenceRequest, List[InferenceRequest]]:
        pass

    @overload
    def get_request(self, timeout: None = None) -> Union[InferenceRequest, List[InferenceRequest]]:
        pass

    @overload
    def get_request(self, timeout: Optional[float] = None) -> Optional[Union[InferenceRequest, List[InferenceRequest]]]:
        pass

    def get_request(self, timeout: Optional[float] = None) -> Optional[Union[InferenceRequest, List[InferenceRequest]]]:
        if timeout is not None and not self.child_conn.poll(timeout):
            return None
        data = self.child_conn.recv()
        if isinstance(data, list):
            return [InferenceRequest.from_dict(req) for req in data]
        return InferenceRequest.from_dict(data)

    def send_response(self, response: Optional[InferenceResponse]):
        self.child_conn.send(response.to_dict() if response else None)
