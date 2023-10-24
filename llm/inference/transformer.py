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
EncoderHiddenState = torch.FloatTensor
PastKeyValues = Tuple[Tuple[torch.FloatTensor, torch.FloatTensor]]
AttentionMask = torch.Tensor


class TransformersEngine(InferenceEngine):
    """
    Generate a token stream from a prompt with the given transformer model
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        batch_size: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

        self.batch: List[InferenceState] = []
        self.pending: List[InferenceState] = []
        self.kv_cache = PastKVCache()
        self.attn_cache = AttentionCache()
        self.encoder_cache = EncoderCache()
        self.encoder_attn_cache = AttentionCache()
        self.has_updates: bool = False

        # Pad token is required
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @classmethod
    def from_model_config(cls, model_config: ModelConfig, load_kwargs: Optional[Dict[str, Any]] = None, **kwargs) -> TransformersEngine:
        model, tokenizer = model_config.load(**(load_kwargs or {}))
        kwargs.setdefault("max_length", model_config.max_length)
        return cls(model, tokenizer, **kwargs)

    def add_requests(self, requests: List[InferenceRequest]):
        states = self.process_inputs(requests)
        self.pending.extend(states)

    @torch.inference_mode()
    def generate_batch_stream(self, requests: List[InferenceRequest]) -> Iterable[InferenceResponse]:
        """
        Generate responses for multiple requests in batches. Responses are streamed individually
        as they recieve updates. Once a response has been marked "stopped", it will no longer be
        yielded.
        """
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
            self._clear()

    @torch.inference_mode()
    def generate_batch(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """
        Generate responses for multiple requests in batches. Responses are returned in the order
        in which they were recieved after all requests have been completed.
        """
        self.add_requests(requests)
        all_states: List[InferenceState] = []
        while self.pending:
            self.update_batch()
            all_states.extend(self.batch)
            self._check_completed()

            while len(self.batch) > 0:
                self.decode_step()
                self._check_completed()
        self._clear()
        return [s.resp for s in all_states]

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
        for response in self.generate_batch_stream([request]):
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
        request = InferenceRequest(
            prompt,
            max_new_tokens=max_new_tokens,
            echo_prompt=echo_prompt,
            stop_token_ids=stop_token_ids,
            stop_strings=stop_strings,
            **kwargs
        )
        return self.generate_batch([request])[0]

    def add_requests(self, requests: List[InferenceRequest]):
        states = self.process_inputs(requests)
        self.pending.extend(states)

    def update_batch(self):
        """
        Pull requests from pending into a new batch, and run the prefill step
        to initialize the attention cache from input tokens.

        Currently only static-batching is supported.
        """
        if len(self.batch) != 0:
            raise Exception("Updating an in-progress batch is not currently supported.")

        num = min(self.batch_size - len(self.batch), len(self.pending))
        new_states = self.pending[:num]
        self.pending = self.pending[num:]
        self.batch.extend(new_states)

        new_tokens = [s.input_ids for s in new_states]
        num_tokens = [len(tokens) for tokens in new_tokens]
        max_length = max(num_tokens)

        padded = self.tokenizer.pad(
            {"input_ids": new_tokens},
            padding="max_length",
            max_length=max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # First step calculates attention of full input, and caches past key values
        logits, past_key_values, encoder_hidden_state = self.prefill(
            padded.input_ids, padded.attention_mask
        )

        # Update caches
        self.kv_cache.append(past_key_values)
        self.encoder_cache.append(encoder_hidden_state)
        if self.model.config.is_encoder_decoder:
            self.encoder_attn_cache.set(padded.attention_mask)
            self.attn_cache.clear()
            self.attn_cache.pad(1, batch_size=self.encoder_attn_cache.batch_size)
        else:
            self.attn_cache.set(padded.attention_mask)
            self.attn_cache.pad(1)

        # Convert initial logits to tokens
        self.process_logits_batch(logits, new_states)

    def decode_step(self):
        """
        Get the next token for each state in the current batch.

        It is assumed that the attention cache has been prefilled for the batch first.
        """
        logits, past_key_values = self.decoder(
            [state.tokens[-1] for state in self.batch],
            past_key_values=self.kv_cache.data,
            attention_mask=self.attn_cache.data,
            encoder_hidden_state=self.encoder_cache.data,
            encoder_attention_mask=self.encoder_attn_cache.data,
        )
        self.kv_cache.set(past_key_values)
        self.attn_cache.pad(1)
        self.process_logits_batch(logits)

    def prefill(
        self, input_ids: torch.Tensor, attention_mask: Optional[AttentionMask] = None
    ) -> Tuple[
        Logits, PastKeyValues, Optional[EncoderHiddenState]
    ]:
        """
        First pass through the model. Attention is calculated for the entire input.
        In subsequent steps, attention is only calculated for new tokens.
        """
        encoder_hidden_state: Optional[EncoderHiddenState] = None
        if self.model.config.is_encoder_decoder:
            # Encoder-Decoder models
            encoder_hidden_state = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            start_ids = torch.as_tensor(
                [[self.model.generation_config.decoder_start_token_id]] * len(input_ids),
                dtype=torch.int64,
                device=self.model.device,
            )
            out = self.model.decoder(
                input_ids=start_ids,
                encoder_hidden_states=encoder_hidden_state,
                use_cache=True,
            )
            logits = self.model.lm_head(out.last_hidden_state)
        else:
            # Decoder-only models
            out = self.model(input_ids, attention_mask=attention_mask, use_cache=True)
            logits = out.logits
        return logits, out.past_key_values, encoder_hidden_state

    def decoder(
        self,
        last_tokens: List[int],
        past_key_values: PastKeyValues,
        attention_mask: Optional[AttentionMask] = None,
        encoder_hidden_state: Optional[EncoderHiddenState] = None,
        encoder_attention_mask: Optional[AttentionMask] = None,
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
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_state,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )
            logits = self.model.lm_head(out.last_hidden_state)
        else:
            # Decoder-only models
            out = self.model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
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
        if not state.resp.stopped and not state.req.stop_strings and stream_interval <= 0:
            # Skip decoding tokens when not needed
            return

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
            stop_pos, partial_stop = check_stop_str(output, state.req.stop_strings, rfind_start)
            if stop_pos != -1:
                output = output[:stop_pos]
                state.set_stopped("stop string")

            # Prevent updating output with partial stop strings
            if not partial_stop or state.resp.stopped:
                state.set_output(output)
                self.has_updates = True

    def process_inputs(self, requests: List[InferenceRequest]) -> List[InferenceState]:
        """
        Prepare requests for inference
        """
        prompts: List[str] = [""] * len(requests)
        input_ids: List[List[int]] = [[]] * len(requests)
        to_tokenize: List[Tuple[i, str]] = []

        # Collect input prompts and tokens
        for i, req in enumerate(requests):
            if req.stop_token_ids is None:
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
        if to_tokenize:
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
        if not self.has_updates:
            return

        for state in self.batch:
            if state.resp.stopped or state.output_updated:
                yield state.resp
                state.output_updated = False

    def _check_completed(self):
        if not self.has_updates:
            return

        # Filter out completed indices
        indices: Set[int] = set()
        for i, state in enumerate(self.batch):
            if state.resp.stopped:
                indices.add(i)

        if indices:
            # Filter out cache values for completed requests
            self.kv_cache.discard(indices)
            self.encoder_cache.discard(indices)
            self.attn_cache.discard(indices)
            self.encoder_attn_cache.discard(indices)
            self.batch = [s for s in self.batch if not s.resp.stopped]

        self.has_updates = False

    def clear_all(self, stopped_reason: str = "cancelled") -> List[InferenceResponse]:
        states = self.batch + self.pending
        self.batch = []
        self.pending = []
        for state in states:
            if not state.resp.stopped:
                state.set_stopped(stopped_reason)
        self._clear()
        return [s.resp for s in states]

    def _clear(self):
        self.kv_cache.clear()
        self.attn_cache.clear()
        self.encoder_cache.clear()
        self.encoder_attn_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()


@dataclass
class InferenceRequest(DataclassBase):
    """
    Input data for inference
    """
    prompt: Union[str, List[int]]

    uid: str = field(default_factory=lambda: uuid4().hex)
    max_new_tokens: int = 256
    stream_interval: int = 2
    echo_prompt: bool = False
    stop_strings: Union[str, List[str]] = ""
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
        is_stop_token = self.req.stop_token_ids and (token in self.req.stop_token_ids)
        if is_stop_token or self.resp.tokens_generated == self.req.max_new_tokens:
            self.set_stopped(
                "stop token" if is_stop_token else "max new tokens"
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
        return self.data[0][0].shape[0]

    @property
    def sequence_length(self) -> int:
        if not self.data:
            return 0
        return self.data[0][0].shape[2]

    def set(self, data: PastKeyValues):
        self.data = data

    def append(self, data: PastKeyValues, dim: int = 0):
        if not self.data:
            self.data = data
            return
        self.data = tuple(
            (torch.cat((k1, k2), dim=0), torch.cat((v1, v2), dim=dim))
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
            self.clear()

    def clear(self):
        del self.data
        self.data = tuple()


class EncoderCache:
    def __init__(self, data: Optional[EncoderHiddenState] = None):
        self.data = data

    @property
    def batch_size(self) -> int:
        if self.data is None:
            return 0
        return self.data.shape[0]

    def set(self, data: Optional[EncoderHiddenState]):
        self.data = data

    def append(self, data: Optional[EncoderHiddenState], dim: int = 0):
        if data is None:
            return
        if self.data is None:
            self.data = data
            return
        self.data = torch.cat((self.data, data), dim=dim)

    def discard(self, indices: Iterable[int]):
        if not indices or self.data is None:
            return
        if not isinstance(indices, set):
            indices = set(indices)
        to_keep = [i for i in range(self.batch_size) if i not in indices]
        if len(to_keep) > 0:
            self.data = self.data[to_keep]
        else:
            self.clear()

    def clear(self):
        del self.data
        self.data = None


class AttentionCache:
    mask: torch.Tensor

    def __init__(self, data: Optional[AttentionMask] = None):
        self.data = data

    @property
    def batch_size(self) -> int:
        if self.data is None:
            return 0
        return self.data.shape[0]

    def set(self, data: Optional[AttentionMask]):
        self.data = data

    def append(self, data: Optional[AttentionMask], dim: int = 0):
        if data is None:
            return
        if self.data is None:
            self.data = data
            return
        self.data = torch.cat((self.data, data), dim=dim)

    def pad(self, value: int, length: int = 1, batch_size: Optional[int] = None):
        if batch_size is None:
            batch_size = self.batch_size
        padding = torch.full((batch_size, length), value)
        self.append(padding, dim=1)

    def discard(self, indices: Iterable[int]):
        if not indices or self.data is None:
            return
        if not isinstance(indices, set):
            indices = set(indices)
        to_keep = [i for i in range(self.batch_size) if i not in indices]
        if len(to_keep) > 0:
            self.data = self.data[to_keep]
        else:
            self.clear()

    def clear(self):
        del self.data
        self.data = None
