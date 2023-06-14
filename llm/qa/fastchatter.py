from queue import Queue
from threading import Thread
from typing import Any, Dict, Iterable, List, Optional

from fastchat.serve.inference import generate_stream
from fastchat.conversation import Conversation

from llm.qa import model_configs
from llm.qa.document_store import DocStore
from llm.qa.prompts import QAPrompt


class StreamRequest:
    """
    A generate_stream request to be enqueued
    """
    def __init__(self, prompt: str, **kwargs) -> None:
        self.prompt = prompt
        self.kwargs = kwargs
        self.output: Queue[Optional[str]] = Queue()


class InferenceEngine:
    """
    Manages request and response queues for thread-safe chat inference
    """

    def __init__(
        self,
        model_config: model_configs.ChatModelConfig = model_configs.VICUNA,
        queue: Optional[Queue[StreamRequest]] = None,
    ):
        self.model_config = model_config
        self.model, self.tokenizer = self.model_config.load()
        self.queue: Queue[StreamRequest] = queue or Queue()
        self.watch_thread: Optional[Thread] = None

    def request(self, prompt: str, **kwargs) -> Iterable[str]:
        """
        Enqueue generation request, and dequeue the output stream until completed
        """
        if not self.watch_thread or not self.watch_thread.is_alive():
            raise Exception("Watch thread is not running")

        request = StreamRequest(prompt, **kwargs)
        self.queue.put(request)
        while True:
            text = request.output.get()
            request.output.task_done()
            if text is None:
                # Stream completed
                break
            else:
                # Yield current output
                yield text

    def watch(self):
        """
        Spawn a thread to watch the request queue and generate responses

        Should only be called from the main thread
        """
        def _watch():
            while True:
                request = self.queue.get()
                for text in self.generate_stream(request.prompt, **request.kwargs):
                    request.output.put(text)

                # None indicates to the requester that the stream is completed
                request.output.put(None)

        if self.watch_thread is not None and self.watch_thread.is_alive():
            raise Exception("Watch already started")
        self.watch_thread = Thread(None, _watch, daemon=True)
        self.watch_thread.start()

    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        max_new_tokens: int = 512,
        echo: bool = False,
        **kwargs,
    ) -> Iterable[str]:
        """
        Stream generated text by token
        """
        gen_params = {
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "echo": echo,
            **kwargs,
        }

        output_stream = generate_stream(
            self.model,
            self.tokenizer,
            gen_params,
            self.model.device,
            context_len=self.model_config.max_length,
        )

        for outputs in output_stream:
            output_text = outputs["text"].strip()
            yield output_text

    def new_conversation(self) -> Conversation:
        return self.model_config.new_conversation()


class QASession:
    def __init__(self, engine: InferenceEngine, docstore: DocStore, prompt: Optional[QAPrompt] = None, debug: bool = False):
        self.engine = engine
        self.docstore = docstore
        self.conv = engine.new_conversation()
        self.prompt = prompt or engine.model_config.default_prompt
        self.debug = debug
        self.results: List[Dict[str, Any]] = []

    def append_question(self, question: str):
        self.conv.append_message(self.conv.roles[0], question)
        self.conv.append_message(self.conv.roles[1], None)

    def update_context(self, question: str, **kwargs):
        self.results = self.docstore.search(question, **kwargs)
        self.set_context([r["text"] for r in self.results])

    def conversation_stream(self, **kwargs) -> Iterable[str]:
        # TODO: Scrolling conv window while keeping context as long as possible
        prompt = self.conv.get_prompt()
        kwargs.setdefault("stop", self.conv.stop_str)
        kwargs.setdefault("stop_token_ids", self.conv.stop_token_ids)
        for output_text in self.engine.request(prompt, **kwargs):
            yield output_text

        self.conv.messages[-1][-1] = output_text.strip()
        if self.debug:
            print('**DEBUG**')
            print(prompt)
            print(output_text)

    def set_context(self, contexts: List[str]):
        context = ""
        for c in contexts:
            text = c.strip()
            context += f"{self.engine.model_config.context_label}: {text}\n\n"

        prompt_kwargs = {}
        if "roles" in self.prompt.inputs:
            prompt_kwargs["roles"] = self.conv.roles
        if "context_label" in self.prompt.inputs:
            prompt_kwargs["context_label"] = self.engine.model_config.context_label

        self.conv.system = self.prompt.render(context=context, **prompt_kwargs)

    def get_history(self) -> str:
        messages = [f'{role}: {"" if message is None else message}' for role, message in self.conv.messages]
        return "\n\n".join(messages)

    def clear(self, keep_results: bool = False):
        self.conv.messages = []
        self.conv.system = ""
        if not keep_results:
            self.results = []
