from __future__ import annotations

from typing import Iterable, List, Optional, Type, Union

from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
import torch

from llm.inference import TransformersEngine, InferenceEngine
from llm.model_configs import ChatModelConfig, bnb_quantization
from llm.prompt import Message, Prompt, Conversation
from llm.qa.prompts import FewShotQA, StandaloneQuestion


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
        prompt: Prompt,
        standalone_question_prompt: Prompt,
        conversation: Optional[Conversation] = None,
        user_label: str = "Question: ",
        assistant_label: str = "Answer: ",
        debug: bool = False,
    ):
        self.engine = engine
        self.vector_store = vector_store
        self.conversation = conversation if conversation else Conversation()
        self.qa_prompt = prompt
        self.standalone_question_prompt = standalone_question_prompt
        self.user_label = user_label
        self.assistant_label = assistant_label
        self.debug = debug
        self.results: List[Document] = []
        self.contexts: List[str] = []

    @classmethod
    def from_model_config(
        cls,
        model_config: ChatModelConfig,
        vector_store: VectorStore,
        engine: Optional[InferenceEngine] = None,
        prompt: Union[Prompt, Type[Prompt]] = FewShotQA,
        standalone_question_prompt: Union[Prompt, Type[Prompt]] = StandaloneQuestion,
        **kwargs,
    ) -> QASession:
        if engine is None:
            engine = TransformersEngine.from_model_config(
                model_config,
                model_kwargs={
                    "device_map": "auto",
                    "torch_dtype": torch.float16,
                    "quantization_config": bnb_quantization(),
                },
            )
        if isinstance(prompt, type):
            prompt = prompt(format=model_config.format)
        if isinstance(standalone_question_prompt, type):
            standalone_question_prompt = standalone_question_prompt(format=model_config.format)
        return cls(engine, vector_store, **kwargs)

    def stream_answer(self, question: str, update_context: bool = False, with_prefix: bool = False, **kwargs) -> Iterable[str]:
        """
        Stream response to the given question using the session's prompt and contexts.
        """
        if update_context:
            self.search_context(question)

        last_message = self.last_message
        if last_message and last_message.input == question and last_message.response is None:
            # Message already added to conversation
            message = self.last_message
            message.contexts = self.contexts
        else:
            message = self.append_question(question, contexts=self.contexts)

        input_text = self.conversation.render(self.qa_prompt)
        if self.debug:
            print(f"\n** Context Input **\n{input_text}")

        gen_kwargs = {
            "stop": self.qa_prompt.format.stop_strings,
            "temperature": 0.7,
            "top_p": 0.9,
            **kwargs,
        }
        prefix = ""
        if with_prefix:
            prefix = self.assistant_label

        output_text = ""
        for output_text in self.engine.generate_stream(input_text, **gen_kwargs):
            output_text = output_text.strip()
            yield prefix + output_text

        message.response = output_text
        if self.debug:
            print(f"\n** Context Answer **\n{output_text}")

    def rephrase_question(self, question: str, **kwargs):
        """
        Rephrase question to be a standalone question based on conversation history.

        Enables users to implicitly refer to previous messages. Relevant information is
        added to the question, which then gets used for semantic search of contexts.
        """
        last_message = self.last_message
        if not last_message:
            # No history to use for rephrasing
            return question

        if last_message.input == question and last_message.response is None:
            # Question already added to conversation
            messages = self.conversation.messages
            if len(messages) == 1:
                # No history to use for rephrasing
                return question
        else:
            messages = [*self.conversation.messages, Message(question)]

        input_text = self.standalone_question_prompt.render(
            messages=messages, with_contexts=False
        )
        if self.debug:
            print(f"\n** Standalone Input **\n{input_text}")

        params = {
            "stop": self.standalone_question_prompt.format.stop_strings,
            "temperature": 0.7,
            "top_p": 0.9,
            **kwargs,
        }
        standalone = self.engine.generate(input_text, **params).strip()
        if self.debug:
            print(f"\n** Standalone Question **\n{standalone}")
        return standalone

    def append_question(self, question: str, **kwargs) -> Message:
        message = Message(input=question, **kwargs)
        self.conversation.add(message)
        return message

    def search_context(self, question: str, top_k: int = 3, **kwargs) -> List[Document]:
        """
        Update contexts from vector store
        """
        self.results = self.vector_store.similarity_search(question, top_k, **kwargs)
        self.set_contexts([r.page_content for r in self.results])
        return self.results

    def set_contexts(self, contexts: List[str]):
        """
        Set contexts explicitly (e.g. for filtering which results are included)
        """
        self.contexts = contexts

    @property
    def has_history(self) -> bool:
        return len(self.conversation.messages) > 0

    @property
    def last_message(self) -> Optional[Message]:
        if self.has_history:
            return self.conversation.messages[-1]
        return None

    def get_history(self, separator: str = "\n") -> str:
        """
        Get conversation history
        """
        history = []
        for message in self.conversation.messages:
            history.append(f"{self.user_label}{message.input}")
            if message.response is not None:
                history.append(f"{self.assistant_label}{message.response}")

        return separator.join(history)

    def clear(self, keep_results: bool = False):
        self.conversation.clear()
        if not keep_results:
            self.results = []
            self.contexts = []
