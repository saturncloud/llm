from __future__ import annotations

from typing import Iterable, List, Optional

from fastchat.conversation import Conversation
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore

from llm.qa.inference import FastchatEngine, InferenceEngine
from llm.qa.model_configs import ChatModelConfig
from llm.qa.prompts import ZERO_SHOT, ContextPrompt


class QASession:
    """
    Manages session state for a question-answering conversation between a user and an AI.
    Contexts relevant to questions are retrieved from the given vector store and appended
    to the system prompt that is fed into inference.
    """
    def __init__(
        self,
        engine: InferenceEngine,
        vector_store: VectorStore,
        conv: Conversation,
        prompt: ContextPrompt = ZERO_SHOT,
        debug: bool = False,
    ):
        self.engine = engine
        self.vector_store = vector_store
        self.conv = conv
        self.prompt = prompt
        self.debug = debug
        self.results: List[Document] = []

    @classmethod
    def from_model_config(
        cls,
        model_config: ChatModelConfig,
        vector_store: VectorStore,
        engine: Optional[InferenceEngine] = None,
        prompt: Optional[ContextPrompt] = None,
        **kwargs,
    ) -> QASession:
        if engine is None:
            model, tokenizer = model_config.load()
            engine = FastchatEngine(model, tokenizer, model_config.max_length)
        conv = model_config.new_conversation()
        return cls(engine, vector_store, conv, prompt or model_config.default_prompt, **kwargs)

    def append_question(self, question: str, update_context: bool = False, **kwargs):
        self.conv.append_message(self.conv.roles[0], question)
        self.conv.append_message(self.conv.roles[1], None)
        if update_context:
            self.update_context(question, **kwargs)

    def update_context(self, question: str, top_k: int = 3, **kwargs):
        self.results = self.vector_store.similarity_search(question, top_k)
        self.set_context([r.page_content for r in self.results], **kwargs)

    def set_context(self, contexts: List[str], **kwargs):
        if "roles" in self.prompt.inputs:
            kwargs.setdefault("roles", self.conv.roles)
        self.conv.system = self.prompt.render(contexts, **kwargs)

    def conversation_stream(self, **kwargs) -> Iterable[str]:
        input_text = self.conv.get_prompt()
        params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.0,
            "max_new_tokens": 512,
            "echo": False,
            "stop": self.conv.stop_str,
            "stop_token_ids": self.conv.stop_token_ids,
            **kwargs,
        }
        for output_text in self.engine.generate_stream(input_text, **params):
            yield output_text

        self.conv.messages[-1][-1] = output_text.strip()
        if self.debug:
            print('**DEBUG**')
            print(input_text)
            print(output_text)

    def get_history(self) -> str:
        messages = [f'{role}: {"" if message is None else message}' for role, message in self.conv.messages]
        return "\n\n".join(messages)

    def clear(self, keep_system: bool = False, keep_results: bool = False):
        self.conv.messages = []
        if not keep_system:
            self.conv.system = ""
        if not keep_results:
            self.results = []
