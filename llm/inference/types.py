from dataclasses import dataclass, field

from typing import List, Optional, Union
from uuid import uuid4

from llm.inference.utils import LogitsProcessorConfig
from llm.utils.types import DataclassBase


@dataclass
class InferenceRequest(DataclassBase):
    """
    Input data for inference
    """

    input: Union[str, List[int]]

    uid: str = field(default_factory=lambda: uuid4().hex)
    max_new_tokens: int = 256
    # Number of tokens to generate before decoding for stop string checks and streaming updates
    token_interval: int = 4
    echo: bool = False
    stop_strings: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    do_sampling: Optional[bool] = None


@dataclass
class InferenceResponse(DataclassBase):
    """
    Ouput data from inference.
    """

    uid: str
    output: str = ""
    stopped: bool = False
    stopped_reason: str = ""
    tokens_generated: int = 0


@dataclass
class InferenceState:
    """
    Stores information about an in-progress inference request/response
    """

    req: InferenceRequest
    input_text: str
    input_ids: List[int]
    output_updated: bool = False

    resp: InferenceResponse = field(init=False)
    tokens: List[int] = field(init=False)
    logits_config: LogitsProcessorConfig = field(init=False)

    def __post_init__(self):
        self.resp = InferenceResponse(self.req.uid)
        if self.req.echo:
            self.tokens = list(self.input_ids)
        else:
            self.tokens = []
        self.logits_config = LogitsProcessorConfig(
            temperature=self.req.temperature,
            top_p=self.req.top_p,
            top_k=self.req.top_k,
            repetition_penalty=self.req.repetition_penalty,
            do_sampling=self.req.do_sampling,
        )

    def add_token(self, token: int):
        self.tokens.append(token)
        self.resp.tokens_generated += 1
        is_stop_token = self.req.stop_token_ids and (token in self.req.stop_token_ids)
        if is_stop_token or self.resp.tokens_generated == self.req.max_new_tokens:
            self.set_stopped("stop token" if is_stop_token else "max new tokens")

    def set_stopped(self, reason: str):
        self.resp.stopped = True
        self.resp.stopped_reason = reason

    def set_output(self, output: str):
        self.resp.output = output
        self.output_updated = True
