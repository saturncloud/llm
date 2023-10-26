from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Iterable, List, Optional, Union


class InferenceEngine(ABC):
    """
    Interface for string prompt completion
    """

    @abstractmethod
    def generate_stream(
        input: str,
        max_new_tokens: int = 256,
        echo: bool = False,
        stop_token_ids: Optional[List[int]] = None,
        stop_strings: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Iterable[str]:
        """
        Stream generated text as new tokens are added
        """
        raise NotImplementedError()

    def generate(
        self,
        input: str,
        max_new_tokens: int = 256,
        echo: bool = False,
        stop_token_ids: Optional[List[int]] = None,
        stop_strings: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> str:
        answer = ""
        for _answer in self.generate_stream(
            input,
            max_new_tokens=max_new_tokens,
            echo=echo,
            stop_token_ids=stop_token_ids,
            stop_strings=stop_strings,
            **kwargs,
        ):
            answer = _answer
        return answer
