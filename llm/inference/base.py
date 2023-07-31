from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Iterable


class InferenceEngine(ABC):
    """
    Interface for prompt completion
    """

    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> Iterable[str]:
        """
        Stream generated text as each new token is added
        """
        raise NotImplementedError()

    def get_answer(self, prompt: str, **kwargs) -> str:
        answer = ""
        for _answer in self.generate_stream(prompt, **kwargs):
            answer = _answer
        return answer
