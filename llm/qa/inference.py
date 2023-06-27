from __future__ import annotations
from abc import ABC, abstractmethod

from queue import Queue
from threading import Thread
from typing import Iterable, Optional

from fastchat.serve.inference import generate_stream
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class InferenceEngine(ABC):
    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> Iterable[str]:
        """
        Stream generated text as each new token is added
        """
        raise NotImplementedError()


class FastchatEngine(InferenceEngine):
    """
    Wrapper on FastChat's generate_stream that implements
    the InferenceEngine interface
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def generate_stream(
        self,
        prompt: str,
        **kwargs,
    ) -> Iterable[str]:
        gen_params = {
            "prompt": prompt,
            **kwargs,
        }

        output_stream = generate_stream(
            self.model,
            self.tokenizer,
            gen_params,
            self.model.device,
            context_len=self.max_length,
        )

        for outputs in output_stream:
            output_text = outputs["text"].strip()
            yield output_text


class QueuedEngine(InferenceEngine):
    """
    Run inference engines in background threads with queued requests/responses

    Enables thread-safe handling of concurrent requests to one or more engines
    """

    def __init__(self, *engines: InferenceEngine):
        self.queue: Queue[StreamRequest] = Queue()
        for engine in engines:
            self.add_watcher(engine)

    def add_watcher(self, engine: InferenceEngine) -> Thread:
        def _watch():
            while True:
                request = self.queue.get()
                for text in engine.generate_stream(request.prompt, **request.kwargs):
                    request.output.put(text)

                # None indicates to the requester that the stream is completed
                request.output.put(None)

        watch_thread = Thread(None, _watch, daemon=True)
        watch_thread.start()
        return watch_thread

    def generate_stream(
        self,
        prompt: str,
        **kwargs,
    ) -> Iterable[str]:
        request = StreamRequest(prompt, **kwargs)
        self.queue.put(request)
        while True:
            text = request.output.get()
            request.output.task_done()
            if text is not None:
                yield text
            else:
                # Stream completed
                break


class StreamRequest:
    """
    An inference request to be enqueued
    """
    def __init__(self, prompt: str, **kwargs) -> None:
        self.prompt = prompt
        self.kwargs = kwargs
        # When retrieved value is None, stream is completed
        self.output: Queue[Optional[str]] = Queue()
