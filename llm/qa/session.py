from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain.schema.messages import AIMessage, BaseMessage, HumanMessage

from llm.inference import TransformersEngine, InferenceEngine
from llm.model_configs import ChatModelConfig
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
        conv: Optional[ConversationBufferWindowMemory] = None,
        prompt: ContextPrompt = ZERO_SHOT,
        human_label: Optional[str] = None,
        ai_label: Optional[str] = None,
        debug: bool = False,
    ):
        self.engine = engine
        self.vector_store = vector_store
        self.conv = conv or ConversationBufferWindowMemory(human_prefix="Question:", ai_prefix="Answer:")
        self.prompt = prompt
        self.human_label = human_label or self.conv.human_prefix
        self.ai_label = ai_label or self.conv.ai_prefix
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
            engine = TransformersEngine(model, tokenizer, model_config.max_length)
        conv = model_config.new_conversation()
        return cls(engine, vector_store, conv, prompt or model_config.default_prompt, **kwargs)

    def stream_answer(self, question: str, update_context: bool = False, with_prefix: bool = False, **kwargs) -> Iterable[str]:
        """
        Stream response to the given question using the session's prompt and contexts.
        """
        last_message = self.last_message
        if isinstance(last_message, AIMessage):
            # Question has not been appended to conversation yet
            self.append_question(question)

        if update_context:
            self.search_context(question)
        prompt_kwargs = {}
        if "roles" in self.prompt.inputs:
            prompt_kwargs["roles"] = self.roles
        input_text = self.prompt.render(question=question, contexts=self.contexts, **prompt_kwargs)

        gen_kwargs = {
            "stop": [self.conv.human_prefix, self.prompt.default_context_label],
            "temperature": 0.7,
            "top_p": 0.9,
            **kwargs,
        }
        prefix = ""
        if with_prefix:
            prefix = self.format_answer("")

        for output_text in self.engine.generate_stream(input_text, **gen_kwargs):
            output_text = output_text.strip()
            yield prefix + output_text

        self.append_answer(output_text)
        if self.debug:
            print(f"\n** Context Input **\n{input_text}")
            print(f"\n** Context Answer **\n{output_text}")

    def rephrase_question(self, question: str, **kwargs):
        """
        Rephrase question to be a standalone question based on conversation history.

        Enables users to implicitly refer to previous messages. Relevant information is
        added to the question, which then gets used both for semantic search the final answer.
        """
        if not self.has_history:
            return question

        range_end: Optional[int] = None
        if isinstance(self.last_message, HumanMessage) and self.last_message.content == question:
            if len(self.conv.buffer) == 1:
                # No history other than the current question
                return question
            range_end = -1

        history = self.get_history(range_end=range_end)
        input_text = STANDALONE_QUESTION.render(
            conversation=history, roles=self.roles, question=question
        )
        params = {
            "stop": self.conv.human_prefix,
            "temperature": 0.7,
            "top_p": 0.9,
            **kwargs,
        }
        standalone = self.engine.generate(input_text, **params).strip()
        if self.debug:
            print(f"\n** Standalone Input **\n{input_text}")
            print(f"\n** Standalone Question **\n{standalone}")
        return standalone

    def append_question(self, question: str):
        self.conv.chat_memory.add_user_message(question)

    def append_answer(self, answer: str):
        last_message = self.last_message
        if isinstance(last_message, AIMessage) and not last_message.content:
            last_message.content = answer
        else:
            self.conv.chat_memory.add_ai_message(answer)

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
    def roles(self) -> Tuple[str, str]:
        return (self.conv.human_prefix, self.conv.ai_prefix)

    @property
    def has_history(self) -> bool:
        return len(self.conv.buffer) > 0

    @property
    def last_message(self) -> Optional[BaseMessage]:
        if self.has_history:
            return self.conv.buffer[-1]
        return None

    def get_history(
        self, separator: str = "\n", range_start: Optional[int] = None, range_end: Optional[int] = None
    ) -> str:
        """
        Get conversation history
        """
        history = []
        if not range_start:
            if self.conv.k <= 0:
                return ""

            range_start = -2 * self.conv.k
            if len(self.conv.buffer) > -range_start:
                first_message = self.conv.buffer[range_start]
                if isinstance(first_message, AIMessage):
                    range_start -= 1

        if range_end:
            messages = self.conv.buffer[range_start:range_end]
        else:
            messages = messages = self.conv.buffer[range_start:]

        for message in messages:
            if isinstance(message, HumanMessage):
                history.append(self.format_question(message.content))
            else:
                history.append(self.format_answer(message.content))

        return separator.join(history)

    def format_question(self, question: str) -> str:
        return f"{self.human_label} {question}"

    def format_answer(self, answer: str) -> str:
        return f"{self.ai_label} {answer}"

    def clear(self, keep_results: bool = False):
        self.conv.clear()
        if not keep_results:
            self.results = []
            self.contexts = []
