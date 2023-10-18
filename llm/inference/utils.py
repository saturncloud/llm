from __future__ import annotations
from dataclasses import dataclass, field

from typing import List, Optional, Tuple, Union

import torch
from transformers import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    TopKLogitsWarper,
)


@dataclass
class LogitsProcessorConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    do_sampling: Optional[bool] = None

    _logit_processors: LogitsProcessorList = field(init=False)

    def __post_init__(self):
        if self.do_sampling is None:
            if self.temperature < 1e-5 or self.top_p < 1e-8:
                self.do_sampling = False
            else:
                self.do_sampling = True
        self._logit_processors = self._load()

    def _load(self) -> LogitsProcessorList:
        processors = []
        if self.temperature != 1.0 and self.temperature != 0.0:
            processors.append(TemperatureLogitsWarper(self.temperature))
        if self.top_p < 1.0:
            processors.append(TopPLogitsWarper(self.top_p))
        if self.top_k > 0:
            processors.append(TopKLogitsWarper(self.top_k))
        if self.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(self.repetition_penalty))
        return LogitsProcessorList(processors)

    def process(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self._logit_processors(input_ids, scores, **kwargs)


def check_stop_str(
    output: str, stop_strings: Union[str, List[str]], rfind_start: int = 0
) -> Tuple[int, bool]:
    """
    Check for string based stopping conditions on the output

    Return lowest index of any stop strings found, and a bool indicating if a partial stop_str
    was found at the end of the output.
    """
    if not stop_strings:
        return -1, False
    if isinstance(stop_strings, str):
        stop_strings = [stop_strings]

    partial_stop = False
    stop_pos = -1
    for stop_str in stop_strings:
        pos = output.rfind(stop_str, rfind_start)
        if pos != -1:
            output = output[:pos]
            if stop_pos == -1 or pos < stop_pos:
                stop_pos = pos
        else:
            # Check for partial stop_str
            for i in range(0, min(len(output), len(stop_str))):
                if stop_str.startswith(output[-i:]):
                    partial_stop = True
                    break

    return stop_pos, partial_stop
