from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Iterable, List, Optional, Union


class InferenceEngine(ABC):
    """
    Interface for prompt completion
    """

    @abstractmethod
    def generate_stream(
        prompt: str,
        max_new_tokens: int = 256,
        echo_prompt: bool = False,
        stop_token_ids: Optional[List[int]] = None,
        stop_strings: Union[str, List[str]] = "",
        **kwargs,
    ) -> Iterable[str]:
        """
        Stream generated text as each new token is added
        """
        raise NotImplementedError()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        echo_prompt: bool = False,
        stop_token_ids: Optional[List[int]] = None,
        stop_strings: Union[str, List[str]] = "",
        **kwargs,
    ) -> str:
        answer = ""
        for _answer in self.generate_stream(
            prompt,
            max_new_tokens=max_new_tokens,
            echo_prompt=echo_prompt,
            stop_token_ids=stop_token_ids,
            stop_strings=stop_strings,
            **kwargs,
        ):
            answer = _answer
        return answer
