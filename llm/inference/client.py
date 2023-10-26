import json
import os
from typing import Any, Dict, Iterable, List, Optional, Union
from requests import Response, Session
from llm.inference.base import InferenceEngine
from llm.inference.utils import check_stop_str


class APIClient(InferenceEngine):
    """
    Client for requesting inference from a remote inference server.
    """

    path: str = "api/inference"
    streaming_delimiter: bytes = b"\n\n"

    def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None):
        self.session = Session()
        if headers:
            self.session.headers.update(headers)
        self.base_url = base_url

    def generate_stream(
        self,
        input: str,
        max_new_tokens: int = 256,
        echo_prompt: bool = False,
        stop_token_ids: Optional[List[int]] = None,
        stop_strings: Union[str, List[str]] = "",
        **kwargs,
    ) -> Iterable[str]:
        return self._request(
            input=input,
            max_new_tokens=max_new_tokens,
            echo_prompt=echo_prompt,
            stop_token_ids=stop_token_ids,
            stop_strings=stop_strings,
            streaming=True,
            **kwargs,
        )

    def generate(
        self,
        input: str,
        max_new_tokens: int = 256,
        echo_prompt: bool = False,
        stop_token_ids: Optional[List[int]] = None,
        stop_strings: Union[str, List[str]] = "",
        **kwargs,
    ) -> str:
        text = ""
        for t in self._request(
            input=input,
            max_new_tokens=max_new_tokens,
            echo_prompt=echo_prompt,
            stop_token_ids=stop_token_ids,
            stop_strings=stop_strings,
            streaming=False,
            **kwargs,
        ):
            text = t
        return text

    def _kwarg_mapping(self) -> Dict[str, Optional[str]]:
        return {}

    def _get_inputs(self, **kwargs) -> Dict[str, Any]:
        mapping = self._kwarg_mapping()
        for k_in, k_out in mapping.items():
            if k_in in kwargs:
                value = kwargs.pop(k_in)
                if k_out is not None:
                    kwargs[k_out] = value
        return kwargs

    def _request(
        self,
        streaming: bool = True,
        **kwargs,
    ) -> Iterable[str]:
        kwargs["streaming"] = streaming
        input_data = self._get_inputs(**kwargs)
        url = os.path.join(self.base_url, self.path)

        response = self.session.post(url, json=input_data, stream=streaming)
        if streaming:
            for line in response.iter_lines(delimiter=self.streaming_delimiter):
                if line:
                    data = self._parse_line(line)
                    if data:
                        yield self._get_answer(data, **kwargs)
        else:
            data = self._parse_line(response.content)
            if data:
                yield self._get_answer(data, **kwargs)
            else:
                # No response
                yield ""

    def _parse_line(self, line: bytes) -> Optional[Dict[str, Any]]:
        # Simplified SSE parsing with JSON data
        if line.startswith(b"data:"):
            line = line[len(b"data:"):]
        if line.startswith(b" "):
            line = line[1:]
        if line.endswith(b"\n\n"):
            line = line[:-2]
        if line == b"[DONE]":
            return None
        return json.loads(line)

    def _get_answer(self, data: Dict[str, Any], **kwargs) -> str:
        return data["output"]


class VLLMClient(APIClient):
    """
    Client for requesting inference from a remote VLLM server.
    """

    path: str = "generate"
    streaming_delimiter: bytes = b"\0"

    def _kwarg_mapping(self) -> Dict[str, Optional[str]]:
        return {
            "input": "prompt",
            "max_new_tokens": "max_tokens",
            "stop_strings": "stop",
            "streaming": "stream",
            # VLLM does not support this
            "echo_prompt": None,
        }

    def _get_inputs(self, **kwargs) -> Dict[str, Any]:
        data = super()._get_inputs(**kwargs)
        if "ignore_eos" not in data:
            stop_token_ids = data.get("stop_token_ids", None)
            if isinstance(stop_token_ids, list) and len(stop_token_ids) == 0:
                # Passing empty list disables EOS stopping
                data["ignore_eos"] = True
        return data

    def _get_answer(self, data: Dict[str, Any], **kwargs) -> str:
        input = kwargs["input"]
        echo_prompt = kwargs.get("echo_prompt", False)
        if echo_prompt:
            rfind_start = len(input)
        else:
            rfind_start = 0

        answer = self._trim_answer(
            input,
            data["text"][0],
            echo_prompt=echo_prompt,
        )
        stop_strings = kwargs.get("stop_strings")
        if stop_strings:
            # Strip off partially completed stop strings from the response
            stop_pos, partial_stop = check_stop_str(answer, stop_strings, rfind_start=rfind_start)
            if partial_stop:
                return answer[:stop_pos]
        return answer

    def _trim_answer(self, input: str, answer: str, echo_prompt: bool = False) -> str:
        if not echo_prompt:
            if answer.startswith(input):
                return answer[len(input) :]
        return answer
