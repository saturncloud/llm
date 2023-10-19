from __future__ import annotations
from dataclasses import dataclass
import gc

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    TopKLogitsWarper,
)

from llm.inference.base import InferenceEngine
from llm.model_configs import ModelConfig


class TransformersEngine(InferenceEngine):
    """
    Generate a token stream from a prompt with the given transformer model
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    @classmethod
    def from_model_config(cls, model_config: ModelConfig, load_kwargs: Optional[Dict[str, Any]] = None, **kwargs) -> TransformersEngine:
        model, tokenizer = model_config.load(**(load_kwargs or {}))
        kwargs.setdefault("max_length", model_config.max_length)
        return cls(model, tokenizer, **kwargs)

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        stream_interval: int = 2,
        echo_prompt: bool = False,
        stop_token_ids: Optional[List[int]] = None,
        stop_strings: Union[str, List[str]] = "",
        **logit_kwargs,
    ) -> Iterable[str]:
        logits_config = LogitsProcessorConfig(**logit_kwargs)
        logits_processor = logits_config.load()
        do_sampling = True if logits_config.do_sampling else False

        if not stop_token_ids:
            if self.tokenizer.eos_token_id is not None:
                stop_token_ids = [self.tokenizer.eos_token_id]
            else:
                stop_token_ids = []

        len_prompt = len(prompt)
        if self.tokenizer.bos_token_id is not None:
            add_special_tokens = not prompt.startswith(self.tokenizer.bos_token)
        else:
            add_special_tokens = True

        input_ids = self.tokenizer(prompt, add_special_tokens=add_special_tokens).input_ids
        output_ids = list(input_ids)

        if self.model.config.is_encoder_decoder:
            max_src_len = self.max_length
        else:
            max_src_len = self.max_length - max_new_tokens - 1
        input_ids = input_ids[-max_src_len:]
        input_echo_len = len(input_ids)

        token: Optional[int] = None
        stopped = False

        # First step calculates attention of full input
        logits, past_key_values, encoder_output = self.prefill(input_ids)
        for i in range(max_new_tokens):
            if token:
                # After the first step, only calculate attention of the new token.
                # Previous tokens are represented by past_key_values
                logits, past_key_values = self.decoder(
                    token, past_key_values, encoder_output=encoder_output
                )

            # Process output
            token = self.process_logits(
                logits, output_ids, logits_processor=logits_processor, do_sampling=do_sampling
            )
            output_ids.append(token)

            # Check token stopping conditions
            if token in stop_token_ids or i == max_new_tokens - 1:
                stopped = True

            if (stream_interval > 0 and i % stream_interval == 0) or stopped:
                # Decode tokens and yield current output
                if echo_prompt:
                    tmp_output_ids = output_ids
                    rfind_start = len_prompt
                else:
                    tmp_output_ids = output_ids[input_echo_len:]
                    rfind_start = 0

                output = self.tokenizer.decode(
                    tmp_output_ids,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )

                # Check string stopping conditions
                stop_pos, partially_stopped = check_stop_str(output, stop_strings, rfind_start)
                if stop_pos != -1:
                    output = output[:stop_pos]
                    stopped = True

                # prevent yielding partially completed stop sequences
                if not partially_stopped or stopped:
                    yield output

            if stopped:
                break

        # clean
        del past_key_values, logits, encoder_output
        gc.collect()
        torch.cuda.empty_cache()

    def prefill(
        self, input_ids: List[int]
    ) -> Tuple[
        torch.FloatTensor, Tuple[Tuple[torch.FloatTensor]], Optional[Tuple[torch.FloatTensor]]
    ]:
        """
        First pass through the model. Attention is calculated for the entire input.
        In subsequent steps, attention is only calculated for new tokens.
        """
        input_tensor = torch.as_tensor([input_ids], device=self.model.device)
        encoder_output: Optional[Tuple[torch.FloatTensor]] = None
        if self.model.config.is_encoder_decoder:
            # Encoder-Decoder models
            encoder_output = self.model.encoder(input_ids=input_tensor)[0]
            start_ids = torch.as_tensor(
                [[self.model.generation_config.decoder_start_token_id]],
                dtype=torch.int64,
                device=self.model.device,
            )
            out = self.model.decoder(
                input_ids=start_ids,
                encoder_hidden_states=encoder_output,
                use_cache=True,
            )
            logits = self.model.lm_head(out[0])
        else:
            # Decoder-only models
            out = self.model(input_tensor, use_cache=True)
            logits = out.logits
        return logits, out.past_key_values, encoder_output

    def decoder(
        self,
        last_token: int,
        past_key_values: Tuple[Tuple[torch.FloatTensor]],
        encoder_output: Optional[Tuple[torch.FloatTensor]] = None,
    ) -> Tuple[torch.FloatTensor, Tuple[Tuple[torch.FloatTensor]]]:
        """
        Run a single decode step on the model. Only calculate attention on the most recent token generated,
        previous token's attention is preserved through past_key_values.
        """
        input_tensor = torch.as_tensor([[last_token]], device=self.model.device)
        if self.model.config.is_encoder_decoder:
            # Encoder-Decoder models
            out = self.model.decoder(
                input_ids=input_tensor,
                encoder_hidden_states=encoder_output,
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = self.model.lm_head(out[0])
        else:
            # Decoder-only models
            out = self.model(
                input_ids=input_tensor,
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = out.logits
        return logits, out.past_key_values

    def process_logits(
        self,
        logits: torch.Tensor,
        output_ids: List[int],
        logits_processor: Optional[LogitsProcessorList] = None,
        do_sampling: bool = True,
    ) -> int:
        """
        Process logits and determine the next token in the sequence.
        """
        output_tensor = torch.as_tensor([output_ids], device=logits.device)
        if logits_processor:
            last_token_logits = logits_processor(output_tensor, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if self.model.device.type == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if do_sampling:
            # Sample token from the distribution
            probs = torch.softmax(last_token_logits, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))
        else:
            # Take most probable token
            token = int(torch.argmax(last_token_logits))
        return token

    def generate(self, prompt: str, **kwargs) -> str:
        # Avoid decoding tokens until the final step
        kwargs.setdefault("stream_interval", -1)
        return super().generate(prompt, **kwargs)


@dataclass
class LogitsProcessorConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    do_sampling: Optional[bool] = None
    logit_processors: Optional[LogitsProcessorList] = None

    def __post_init__(self):
        if self.do_sampling is None:
            if self.temperature < 1e-5 or self.top_p < 1e-8:
                self.do_sampling = False
            else:
                self.do_sampling = True

    def load(self) -> LogitsProcessorList:
        processors = []
        if self.temperature != 1.0 and self.temperature != 0.0:
            processors.append(TemperatureLogitsWarper(self.temperature))
        if self.top_p < 1.0:
            processors.append(TopPLogitsWarper(self.top_p))
        if self.top_k > 0:
            processors.append(TopKLogitsWarper(self.top_k))
        if self.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(self.repetition_penalty))
        if self.logit_processors:
            processors.extend(self.logit_processors)
        return LogitsProcessorList(processors)


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
