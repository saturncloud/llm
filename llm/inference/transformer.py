from __future__ import annotations
from dataclasses import dataclass, field
import gc

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from uuid import uuid4

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from llm.inference.base import InferenceEngine
from llm.inference.utils import DataclassBase, LogitsProcessorConfig, check_stop_str
from llm.model_configs import ModelConfig

Logits = torch.FloatTensor
EncoderHiddenState = Tuple[torch.FloatTensor]
PastKeyValues = Tuple[Tuple[torch.FloatTensor, torch.FloatTensor]]


class TransformersEngine(InferenceEngine):
    """
    Generate a token stream from a prompt with the given transformer model
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        batch_size: int = 4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

        self.batch: List[InferenceState] = []
        self.pending: List[InferenceState] = []
        self.kv_cache = PastKVCache()
        self.encoder_cache = EncoderCache()

        # Pad token is required
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

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
        echo_prompt: bool = False,
        stop_token_ids: Optional[List[int]] = None,
        stop_strings: Union[str, List[str]] = "",
        stream_interval: int = 2,
        **kwargs,
    ) -> Iterable[str]:
        """
        Single-prompt inference streaming wrapper. Not safe to use concurrently.
        """
        request = InferenceRequest(
            prompt,
            max_new_tokens=max_new_tokens,
            echo_prompt=echo_prompt,
            stop_token_ids=stop_token_ids,
            stop_strings=stop_strings,
            stream_interval=stream_interval,
            **kwargs
        )
        for response in self.run_batch([request]):
            yield response.output

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        echo_prompt: bool = False,
        stop_token_ids: Optional[List[int]] = None,
        stop_strings: Union[str, List[str]] = "",
        **kwargs,
    ) -> str:
        """
        Single-prompt inference wrapper. Not safe to use concurrently.
        """
        # Longer default stream interval to avoid token-decoding overhead,
        # while still checking periodically for stop strings
        kwargs.setdefault("stream_interval", 20)
        return super().generate(
            prompt,
            max_new_tokens=max_new_tokens,
            echo_prompt=echo_prompt,
            stop_token_ids=stop_token_ids,
            stop_strings=stop_strings,
            **kwargs,
        )

    def add_requests(self, requests: List[InferenceRequest]):
        states = self.process_inputs(requests)
        self.pending.extend(states)

    def run_batch(self, requests: List[InferenceRequest]) -> Iterable[InferenceResponse]:
        try:
            self.add_requests(requests)
            while self.pending:
                self.update_batch()
                yield from self._iterate_updates()
                self._check_completed()

                while len(self.batch) > 0:
                    self.decode_step()
                    yield from self._iterate_updates()
                    self._check_completed()
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    def update_batch(self):
        if len(self.batch) != 0:
            raise Exception("Updating an in-progress batch is not currently supported.")

        num = min(self.batch_size - len(self.batch), len(self.pending))
        new_states = self.pending[:num]
        self.pending = self.pending[num:]
        self.batch.extend(new_states)

        new_tokens = [s.input_ids for s in new_states]
        max_length = len(max(new_tokens))

        padded = self.tokenizer.pad(
            {"input_ids": new_tokens},
            padding="max_length",
            max_length=max_length,
            return_attention_mask=True,
        )

        # First step calculates attention of full input, and caches past key values
        logits, past_key_values, encoder_output = self.prefill(
            padded.input_ids, padded.attention_mask
        )

        # Update caches
        self.kv_cache.append(past_key_values)
        self.encoder_cache.append(encoder_output)

        # Convert initial logits to tokens
        self.process_logits_batch(logits, new_states)

    @torch.inference_mode()
    def decode_step(self):
        # After the first step, only calculate attention of the new token.
        # Previous tokens are represented by the KV cache
        logits, past_key_values = self.decoder(
            [state.tokens[-1] for state in self.batch], self.kv_cache.data, encoder_output=self.encoder_cache.data
        )
        self.kv_cache.set(past_key_values)
        self.process_logits_batch(logits)

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
        return logits, out.past_key_values, encoder_output

    def decoder(
        self,
        last_tokens: List[int],
        past_key_values: PastKeyValues,
        encoder_output: Optional[EncoderHiddenState] = None,
    ) -> Tuple[Logits, PastKeyValues]:
        """
        Run a single model decoder step. Only calculate attention on the most recent token generated,
        previous token's attention is preserved through past_key_values.
        """
        input_tensor = torch.as_tensor([[t] for t in last_tokens], device=self.model.device)
        if self.model.config.is_encoder_decoder:
            # Encoder-Decoder models
            out = self.model.decoder(
                input_ids=input_tensor,
                encoder_hidden_states=encoder_output,
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = self.model.lm_head(out)
        else:
            # Decoder-only models
            out = self.model(
                input_ids=input_tensor,
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = out.logits
        return logits, out.past_key_values

    def process_logits_batch(self, logits: Logits, batch: Optional[List[InferenceState]] = None):
        """
        Process logits and evaluate stopping conditions for all states in the batch
        """
        for i, state in enumerate(batch or self.batch):
            token = self.process_logits(
                logits[i],
                state.tokens,
                logits_config=state.logits_config,
            )
            state.add_token(token)
            self.process_output(state)

    def process_logits(
        self,
        logits: Logits,
        output_ids: List[int],
        logits_config: Optional[LogitsProcessorConfig] = None,
    ) -> int:
        """
        Process logits and determine the next token in the sequence.
        """
        output_tensor = torch.as_tensor([output_ids], device=logits.device)
        if logits_config:
            last_token_logits = logits_config.process(output_tensor, logits[-1, :].unsqueeze(0))[0]
        else:
            last_token_logits = logits[-1, :]

        if self.model.device.type == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if logits_config and logits_config.do_sampling:
            # Sample token from the distribution
            probs = torch.softmax(last_token_logits, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))
        else:
            # Take most probable token
            token = int(torch.argmax(last_token_logits))
        return token

    def process_output(self, state: InferenceState):
        """
        Decode tokens and check for stopping conditions
        """
        stream_interval = state.req.stream_interval
        if (stream_interval > 0 and state.resp.tokens_generated % stream_interval == 0) or state.resp.stopped:
            # Decode tokens, and check if an update needs to be yielded
            if state.req.echo_prompt:
                rfind_start = len(state.prompt)
            else:
                rfind_start = 0

            output = self.tokenizer.decode(
                state.tokens,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )

            # Check string stopping conditions
            stop_pos, partial_stop = check_stop_str(output, state.req.stop, rfind_start)
            if stop_pos != -1:
                output = output[:stop_pos]
                state.set_stopped("stop string")

            # Prevent updating output with partial stop strings
            if not partial_stop or state.resp.stopped:
                state.set_output(output)

    def process_inputs(self, requests: List[InferenceRequest]) -> List[InferenceState]:
        """
        Prepare requests for inference
        """
        prompts: List[str] = [""] * len(requests)
        input_ids: List[List[int]] = [[]] * len(requests)
        to_tokenize: List[Tuple[i, str]] = []

        # Collect input prompts and tokens
        for i, req in enumerate(requests):
            if not req.stop_token_ids:
                if self.tokenizer.eos_token_id is not None:
                    req.stop_token_ids = [self.tokenizer.eos_token_id]
            if isinstance(req.prompt, str):
                to_tokenize.append((i, req.prompt))
                prompts[i] = req.prompt
            else:
                input_ids[i] = req.prompt
                prompts[i] = self.tokenizer.decode(
                    req.prompt,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )

        # Tokenize str prompts
        inputs = self.tokenizer(
            [prompt for _, prompt in to_tokenize],
            padding=False,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        for tokens, (i, _) in zip(inputs.input_ids, to_tokenize):
            input_ids[i] = tokens

        # Collect states
        states: List[InferenceState] = []
        for prompt, tokens, req in zip(prompts, input_ids, requests):
            state = InferenceState(req, prompt, tokens)
            if len(tokens) + req.max_new_tokens >= self.max_length:
                state.set_stopped("input too long")
            states.append(state)

        return states

    def _iterate_updates(self) -> Iterable[InferenceResponse]:
        for state in self.batch:
            if state.resp.stopped or state.output_updated:
                yield state.resp
                state.output_updated = False

    def _check_completed(self):
        # Filter out completed indices
        indices: Set[int] = set()
        for i, state in enumerate(self.batch):
            if state.resp.stopped:
                indices.add(i)

        if indices:
            # Filter out cache values for completed requests
            self.kv_cache.discard(indices)
            self.encoder_cache.discard(indices)
            self.batch = [s for s in self.batch if not s.resp.stopped]


@dataclass
class InferenceRequest(DataclassBase):
    prompt: Union[str, List[int]]

    max_new_tokens: int = 256
    stream_interval: int = 2
    echo_prompt: bool = False
    stop: Union[str, List[str]] = ""
    stop_token_ids: List[int] = field(default_factory=list)

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    do_sampling: Optional[bool] = None

    uid: str = field(init=False, default_factory=lambda: uuid4().hex)


@dataclass
class InferenceResponse(DataclassBase):
    uid: str
    output: str = ""
    stopped: bool = False
    stopped_reason: str = ""
    tokens_generated: int = 0


@dataclass
class InferenceState:
    req: InferenceRequest
    prompt: str
    input_ids: List[int]
    output_updated: bool = False

    resp: InferenceResponse = field(init=False)
    tokens: List[int] = field(init=False)
    logits_config: LogitsProcessorConfig = field(init=False)

    def __post_init__(self):
        self.resp = InferenceResponse(self.req.uid)
        if self.req.echo_prompt:
            self.tokens = list(self.input_ids)
        else:
            self.tokens = []
        self.logits_config = LogitsProcessorConfig(
            temperature=self.req.temperature,
            top_p=self.req.top_p,
            top_k = self.req.top_k,
            repetition_penalty=self.req.repetition_penalty,
            do_sampling=self.req.do_sampling,
        )

    def add_token(self, token: int):
        self.tokens.append(token)
        self.resp.tokens_generated += 1
        if token in self.req.stop_token_ids or self.resp.tokens_generated == self.req.max_new_tokens:
            self.set_stopped(
                "max tokens"
                if self.resp.tokens_generated == self.req.max_new_tokens
                else "stop token"
            )

    def set_stopped(self, reason: str):
        self.resp.stopped = True
        self.resp.stopped_reason = reason

    def set_output(self, output: str):
        self.resp.output = output
        self.output_updated = True


class PastKVCache:
    def __init__(self, data: Optional[PastKeyValues] = None):
        self.data: PastKeyValues = data or tuple()

    @property
    def batch_size(self) -> int:
        if not self.data:
            return 0
        return len(self.data[0][0])

    @property
    def sequence_length(self) -> int:
        if not self.data:
            return 0
        return self.data[0][0].shape[2]

    def set(self, data: PastKeyValues):
        self.data = data

    def append(self, data: PastKeyValues):
        if not self.data:
            self.data = data
            return
        # TODO: Validate shape for EncoderDecoder models
        self.data = tuple(
            (torch.cat((k1, k2), dim=0), torch.cat((v1, v2), dim=0))
            for (k1, v1), (k2, v2) in zip(self.data, data)
        )

    def discard(self, indices: Iterable[int]):
        if not indices:
            return
        if not isinstance(indices, set):
            indices = set(indices)
        to_keep = [i for i in range(self.batch_size) if i not in indices]

        if len(to_keep) > 0:
            self.data = tuple(
                (keys[to_keep], vals[to_keep])
                for keys, vals in self.data
            )
        else:
            self.data = tuple()


class EncoderCache:
    def __init__(self, data: Optional[EncoderHiddenState] = None):
        self.data = data

    @property
    def batch_size(self) -> int:
        if not self.data:
            return 0
        return len(self.data[0])

    def set(self, data: Optional[EncoderHiddenState]):
        self.data = data

    def append(self, data: Optional[EncoderHiddenState]):
        if not data:
            return
        if not self.data:
            self.data = data
            return

        # TODO: Validate shape of encoder data has batch at dim 0
        self.data = tuple(
            torch.cat(s1, s2, dim=0)
            for s1, s2 in zip(self.data, data)
        )

    def discard(self, indices: Iterable[int]):
        if not indices or not self.data:
            return
        if not isinstance(indices, set):
            indices = set(indices)
        to_keep = [i for i in range(self.batch_size) if i not in indices]
        if len(to_keep) > 0:
            self.data = tuple(
                (hidden_states[to_keep])
                for hidden_states in self.data
            )
        else:
            self.data = None
