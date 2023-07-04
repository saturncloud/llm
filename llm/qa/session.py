from __future__ import annotations

from typing import Iterable, List, Optional

from fastchat.conversation import Conversation
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore

from llm.qa.inference import FastchatEngine, InferenceEngine
from llm.qa.model_configs import ChatModelConfig
from llm.qa.prompts import STANDALONE_QUESTION, ZERO_SHOT, ContextPrompt


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
        conv_history: int = 5,
        debug: bool = False,
    ):
        self.engine = engine
        self.vector_store = vector_store
        self.conv = conv
        self.conv_history = conv_history
        self.prompt = prompt
        self.debug = debug
        self.results: List[Document] = []
        self.contexts: List[str] = []

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

    def stream_answer(self, question: str, update_context: bool = False, **kwargs):
        last_message = self.conv.messages[-1] if self.conv.messages else [None, None]
        if last_message[0] == self.conv.roles[1] and last_message[0] is not None:
            # Question has not been appended to conversation
            self.append_question(question)

        if update_context:
            self.search_context(question)
        input_text = self.prompt.render(question=question, contexts=self.contexts)

        params = {
            "stop": self.conv.stop_str,
            "stop_token_ids": self.conv.stop_token_ids,
            **kwargs,
        }
        for output_text in self.engine.generate_stream(input_text, **params):
            yield output_text

        self.conv.update_last_message(output_text.strip())
        if self.debug:
            print(f"\n** Context Input **\n{input_text}")
            print(f"\n** Context Answer **\n{output_text}")

    def rephrase_question(self, question: str, **kwargs):
        if len(self.conv.messages) == 0:
            return question
        input_text = STANDALONE_QUESTION.render(conversation=self.get_history(), question=question)
        params = {
            "stop": self.conv.stop_str,
            "stop_token_ids": self.conv.stop_token_ids,
            **kwargs,
        }
        standalone = self.engine.get_answer(input_text, **params)
        if self.debug:
            print(f"\n** Standalone Input **\n{input_text}")
            print(f"\n** Standalone Question **\n{standalone}")
        return standalone

    def append_question(self, question: str):
        if self.conv_history >= 0:
            keep_messages = self.conv_history * len(self.conv.roles)
            num_messages = len(self.conv.messages)
            if num_messages > keep_messages:
                self.conv.messages = self.conv.messages[num_messages - keep_messages:]
        self.conv.append_message(self.conv.roles[0], question)
        self.conv.append_message(self.conv.roles[1], None)

    def search_context(self, question: str, top_k: int = 3, **kwargs) -> List[Document]:
        self.results = self.vector_store.similarity_search(question, top_k)
        self.set_context([r.page_content for r in self.results], **kwargs)
        return self.results

    def set_context(self, contexts: List[str], **kwargs):
        if "roles" in self.prompt.inputs:
            kwargs.setdefault("roles", self.conv.roles)
        self.contexts = contexts

    def get_history(self, separator: str = "\n", next_question: Optional[str] = None) -> str:
        messages = [f'{role}: {"" if message is None else message}' for role, message in self.conv.messages]
        if next_question:
            messages.append(f"{self.conv.roles[0]}: {next_question}")
        return separator.join(messages)

    def clear(self, keep_system: bool = False, keep_results: bool = False):
        self.conv.messages = []
        if not keep_system:
            self.conv.system = ""
        if not keep_results:
            self.results = []
