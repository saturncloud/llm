from typing import Any, Dict, Iterable, List, Optional
from threading import Lock

from fastchat.serve.inference import generate_stream
from fastchat.conversation import Conversation
import torch

from llm.qa import model_configs
from llm.qa.document_store import DocStore, DEFAULT_INDEX_NAME
from llm.qa.embedding import QAEmbeddings
from llm.qa.prompts import QAPrompt


class MultithreadChat:
    def __init__(
        self,
        model_config: model_configs.ChatModelConfig = model_configs.VICUNA,
    ):
        self.model_config = model_config
        self.model, self.tokenizer = self.model_config.load()
        self.lock = Lock()

    def new_conversation(self) -> Conversation:
        return self.model_config.new_conversation()

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
        gen_params = {
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "echo": echo,
            **kwargs,
        }

        with self.lock:
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


class QASession:
    def __init__(self, chat: MultithreadChat, docstore: DocStore, prompt: Optional[QAPrompt] = None, debug: bool = False):
        self.chat = chat
        self.docstore = docstore
        self.conv = chat.new_conversation()
        self.prompt = prompt or chat.model_config.default_prompt
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
        for output_text in self.chat.generate_stream(prompt, **kwargs):
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
            context += f"{self.chat.model_config.context_label}: {text}\n\n"

        prompt_kwargs = {}
        if "roles" in self.prompt.inputs:
            prompt_kwargs["roles"] = self.conv.roles
        if "context_label" in self.prompt.inputs:
            prompt_kwargs["context_label"] = self.chat.model_config.context_label

        self.conv.system = self.prompt.render(context=context, **prompt_kwargs)

    def get_history(self) -> str:
        messages = [f'{role}: {"" if message is None else message}' for role, message in self.conv.messages]
        return "\n\n".join(messages)

    def clear(self, keep_results: bool = False):
        self.conv.messages = []
        self.conv.system = ""
        if not keep_results:
            self.results = []
