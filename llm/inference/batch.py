from __future__ import annotations
from dataclasses import dataclass, field
import gc

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    LogitsProcessorList,
)

from llm.model_configs import ModelConfig
from llm.inference.transformer import LogitsProcessorConfig, check_stop_str

Logits = torch.FloatTensor
EncoderHiddenState = Tuple[torch.FloatTensor]


@dataclass
class InferenceRequest:
    prompt: str
    max_new_tokens: int = 256
    stream_interval: int = 2
    echo_prompt: bool = False
    stop: Union[str, List[str]] = ""
    stop_token_ids: List[int] = field(default_factory=list)
    logit_kwargs: Dict[str, Any] = field(default_factory=dict)

    input_ids: List[int] = field(init=False)
    attention_mask: List[int] = field(init=False)
    logits_config: LogitsProcessorConfig = field(init=False)
    logits_processor: LogitsProcessorList = field(init=False)

    def __post_init__(self):
        self.logits_config = LogitsProcessorConfig(**self.logit_kwargs)
        self.logits_processor = self.logits_config.load()


@dataclass
class InferenceState:
    request: InferenceRequest
    num_generated: int = 0
    output: str = ""
    stopped: bool = False
    stopped_reason: str = ""

    tokens: List[int] = field(init=False)
    input_len: int = field(init=False)

    def __post_init__(self):
        self.input_len = len(self.request.input_ids)
        if self.request.echo_prompt:
            self.tokens = list(self.request.input_ids)
        else:
            self.tokens = []

    def add_token(self, token: int):
        self.tokens.append(token)
        self.num_generated += 1
        if token in self.request.stop_token_ids or self.num_generated == self.request.max_new_tokens:
            self.set_stopped(
                "max tokens"
                if self.num_generated == self.request.max_new_tokens
                else "stop token"
            )

    def set_stopped(self, reason: str):
        self.stopped = True
        self.stopped_reason = reason


class PastKeyValues:
    def __init__(self, cache: Tuple[Tuple[torch.FloatTensor, torch.FloatTensor]]):
        self.cache = cache

    def keep(self, indices: List[int]):
        self.cache = tuple(
            (keys[indices], values[indices])
            for keys, values in self.cache
        )

    def update(self, cache: Tuple[Tuple[torch.FloatTensor, torch.FloatTensor]]):
        self.cache = cache

    def append(self, cache: Tuple[Tuple[torch.FloatTensor, torch.FloatTensor]]):
        self.cache = tuple(
            (torch.cat((k1, k2), dim=0), torch.cat((v1, v2), dim=0))
            for (k1, v1), (k2, v2) in zip(self.cache, cache)
        )


class BatchInference:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pad token is required
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @classmethod
    def from_model_config(cls, model_config: ModelConfig, **kwargs) -> BatchInference:
        model, tokenizer = model_config.load(**kwargs)
        return cls(model, tokenizer, max_length=model_config.max_length)

    @torch.inference_mode()
    def batch_requests(self, requests: List[InferenceRequest]) -> Iterable[List[InferenceState]]:
        batch_state = self.process_inputs(requests)
        incomplete = [i for (i, state) in enumerate(batch_state) if not state.stopped]

        # First step calculates attention of full input
        logits, past_key_values, encoder_output = self.prefill(
            [requests[i].input_ids for i in incomplete], [requests[i].attention_mask for i in incomplete]
        )

        latest_tokens: List[int] = []
        while incomplete:
            if latest_tokens:
                # After the first step, only calculate attention of the new token.
                # Previous tokens are represented by past_key_values
                logits = self.decoder(
                    latest_tokens, past_key_values, encoder_output=encoder_output
                )

            # Process logits and evaluate stopping conditions
            latest_tokens = []
            do_yield = False
            for i, batch_idx in enumerate(incomplete):
                state = batch_state[batch_idx]
                token = self.process_logits(
                    logits[i],
                    state.tokens,
                    logits_processor=state.request.logits_processor,
                    do_sampling=state.request.logits_config.do_sampling,
                )
                state.add_token(token)
                if self.process_output(state):
                    do_yield = True

                if not state.stopped:
                    latest_tokens.append(token)

            if do_yield:
                yield batch_state

            # Filter out completed indices
            current = incomplete
            incomplete: List[int] = []
            cache_indices: List[int] = []
            for i, batch_idx in enumerate(current):
                if not batch_state[batch_idx].stopped:
                    incomplete.append(batch_idx)
                    cache_indices.append(i)

            # Filter out KV cache from completed requests
            if cache_indices:
                past_key_values.keep(cache_indices)

        # Clear out memory
        del past_key_values, logits, encoder_output
        gc.collect()
        torch.cuda.empty_cache()

    def prefill(
        self, input_ids: List[List[int]], attention_mask: Optional[List[List[int]]] = None
    ) -> Tuple[
        Logits, PastKeyValues, Optional[EncoderHiddenState]
    ]:
        """
        First pass through the model. Attention is calculated for the entire input.
        In subsequent steps, attention is only calculated for new tokens.
        """
        input_tensor = torch.as_tensor(input_ids, device=self.model.device)
        attention_tensor = torch.as_tensor(attention_mask, device=self.model.device) if attention_mask else None
        encoder_output: Optional[EncoderHiddenState] = None
        if self.model.config.is_encoder_decoder:
            # Encoder-Decoder models
            encoder_output = self.model.encoder(input_ids=input_tensor, attention_mask=attention_tensor)
            start_ids = torch.as_tensor(
                [[self.model.generation_config.decoder_start_token_id]] * len(input_ids),
                dtype=torch.int64,
                device=self.model.device,
            )
            out = self.model.decoder(
                input_ids=start_ids,
                encoder_hidden_states=encoder_output,
                use_cache=True,
            )
            logits = self.model.lm_head(out)
        else:
            # Decoder-only models
            out = self.model(input_tensor, attention_mask=attention_tensor, use_cache=True)
            logits = out.logits
        return logits, PastKeyValues(out.past_key_values), encoder_output

    def decoder(
        self,
        last_tokens: List[int],
        past_key_values: PastKeyValues,
        encoder_output: Optional[EncoderHiddenState] = None,
    ) -> Logits:
        """
        Run a single decode step on the model. Only calculate attention on the most recent token generated,
        previous token's attention is preserved through past_key_values.
        """
        input_tensor = torch.as_tensor([[t] for t in last_tokens], device=self.model.device)
        if self.model.config.is_encoder_decoder:
            # Encoder-Decoder models
            out = self.model.decoder(
                input_ids=input_tensor,
                encoder_hidden_states=encoder_output,
                use_cache=True,
                past_key_values=past_key_values.cache,
            )
            logits = self.model.lm_head(out)
        else:
            # Decoder-only models
            out = self.model(
                input_ids=input_tensor,
                use_cache=True,
                past_key_values=past_key_values.cache,
            )
            logits = out.logits
        past_key_values.update(out.past_key_values)
        return logits

    def process_logits(
        self,
        logits: Logits,
        output_ids: List[int],
        logits_processor: Optional[LogitsProcessorList] = None,
        do_sampling: bool = True,
    ) -> int:
        """
        Process logits and determine the next token in the sequence.
        """
        output_tensor = torch.as_tensor([output_ids], device=logits.device)
        if logits_processor:
            last_token_logits = logits_processor(output_tensor, logits[-1, :].unsqueeze(0))[0]
        else:
            last_token_logits = logits[-1, :]

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

    def process_output(self, state: InferenceState) -> bool:
        stream_interval = state.request.stream_interval
        if (stream_interval > 0 and state.num_generated % stream_interval == 0) or state.stopped:
            # Decode tokens, and check if an update needs to be yielded
            if state.request.echo_prompt:
                rfind_start = len(state.request.prompt)
            else:
                rfind_start = 0

            output = self.tokenizer.decode(
                state.tokens,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

            # Check string stopping conditions
            stop_pos, partial_stop = check_stop_str(output, state.request.stop, rfind_start)
            if stop_pos != -1:
                output = output[:stop_pos]
                state.set_stopped("stop string")

            # Prevent updating output with partial stop strings
            if not partial_stop or state.stopped:
                state.output = output
                return True
        return False

    def process_inputs(self, requests: List[InferenceRequest]) -> List[InferenceState]:
        for req in requests:
            if not req.stop_token_ids:
                if self.tokenizer.eos_token_id is not None:
                    req.stop_token_ids = [self.tokenizer.eos_token_id]

        # Tokenize and collect inputs
        inputs = self.tokenizer(
            [req.prompt for req in requests],
            padding="longest",
            add_special_tokens=False,
            return_attention_mask=True,
        )
        batch_state: List[InferenceState] = []
        for i, req in enumerate(requests):
            req.input_ids = inputs.input_ids[i]
            req.attention_mask = inputs.attention_mask[i]

            state = InferenceState(req)
            if len(req.input_ids) + req.max_new_tokens >= self.max_length:
                state.set_stopped("input too long")
            batch_state.append(state)
        return batch_state
