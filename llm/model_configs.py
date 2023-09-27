from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    __version__ as TRANSFORMERS_VERSION,
)

from llm.utils.data import merge_dict
from llm.prompt import ChatMLFormat, DollyFormat, Llama2Format, Conversation, PromptFormat, RedpajamaFormat, TogetherLlama2Format, VicunaFormat

default_model_kwargs = {}
default_tokenizer_kwargs = {
    # https://github.com/huggingface/transformers/pull/24565
    "legacy": False,
}
default_conversation_kwargs = {}

_registry: Dict[str, ModelConfig] = {}


def bnb_quantization() -> BitsAndBytesConfig:
    """
    Create a valid BitsAndBytes quantization for the version of transformers that is installed.
    """
    if TRANSFORMERS_VERSION >= "4.30.0":
        # 4-bit supported after this version
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    return BitsAndBytesConfig(
        load_in_8bit=True,
    )


@dataclass
class ModelConfig:
    """
    Stores model and tokenizer configuration for
    pretrained huggingface models.
    """
    model_id: str
    max_length: int = 512
    format: PromptFormat = field(default_factory=PromptFormat)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_cls: Optional[Type[PreTrainedModel]] = None
    tokenizer_cls: Optional[Type[PreTrainedTokenizerBase]] = None
    peft_adapter: Optional[str] = None
    peft_kwargs: Dict[str, Any] = field(default_factory=dict)

    merge_defaults: bool = True

    def __post_init__(self):
        if self.merge_defaults:
            self.model_kwargs = merge_dict(self.model_kwargs, default_model_kwargs)
            self.tokenizer_kwargs = merge_dict(self.tokenizer_kwargs, default_tokenizer_kwargs)

        self.model_id = trim_model_path(self.model_id)
        if self.peft_adapter:
            self.peft_adapter = trim_model_path(self.peft_adapter)

        _registry[self.name] = self

    @classmethod
    def from_registry(cls, name: str) -> ModelConfig:
        if name not in _registry:
            name = trim_model_path(name)
        if name in _registry:
            return _registry[name]
        logging.warn(f'ModelConfig "{name}" not found in registry. Using generic configuration.')
        return cls(name)

    @property
    def name(self):
        if self.peft_adapter:
            return self.peft_adapter
        return self.model_id

    def load(
        self,
        device_map: Optional[Union[str, Dict]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        model = self.load_model(device_map=device_map, **(model_kwargs or {}))
        tokenizer = self.load_tokenizer(**(tokenizer_kwargs or {}))
        return model, tokenizer

    def load_model(self, **kwargs) -> PreTrainedModel:
        model_cls = self.model_cls or AutoModelForCausalLM
        model_kwargs = {
            **self.model_kwargs,
            **kwargs,
        }
        model = model_cls.from_pretrained(self.model_id, **model_kwargs)
        if self.peft_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.peft_adapter, **self.peft_kwargs)
        return model

    def load_tokenizer(self, **kwargs) -> PreTrainedTokenizerBase:
        tokenizer_cls = self.tokenizer_cls or AutoTokenizer
        tokenizer_kwargs = {
            **self.tokenizer_kwargs,
            **kwargs,
        }
        return tokenizer_cls.from_pretrained(self.model_id, **tokenizer_kwargs)


def trim_model_path(model_id: str) -> str:
    """
    Strip trailing / from path-based model IDs for consistency.
    """
    if os.path.isdir(model_id) and model_id.endswith("/"):
        return model_id.rstrip("/")
    return model_id


VICUNA_7B = ModelConfig(
    "lmsys/vicuna-7b-v1.5",
    max_length=4096,
    format=VicunaFormat(),
)

VICUNA_13B = ModelConfig(
    "lmsys/vicuna-13b-v1.5",
    max_length=4096,
    format=VicunaFormat(),
)

VICUNA_33B = ModelConfig(
    "lmsys/vicuna-33b-v1.5",
    max_length=4096,
    format=VicunaFormat(),
)

LLAMA2_7B = ModelConfig(
    "meta-llama/Llama-2-7b-chat-hf",
    max_length=4096,
    format=Llama2Format(),
)

LLAMA2_13B = ModelConfig(
    "meta-llama/Llama-2-13b-chat-hf",
    max_length=4096,
    format=Llama2Format(),
)

LLAMA2_7B_32K = ModelConfig(
    "togethercomputer/LLaMA-2-7B-32K",
    max_length=32768,
    format=TogetherLlama2Format(),
)

LLAMA2_7B_32K_INSTRUCT = ModelConfig(
    "togethercomputer/Llama-2-7B-32K-Instruct",
    max_length=32768,
    format=TogetherLlama2Format(),
)

REDPAJAMA_7B_INSTRUCT = ModelConfig(
    "togethercomputer/RedPajama-INCITE-7B-Instruct",
    format=RedpajamaFormat(),
)

REDPAJAMA_7B_CHAT = ModelConfig(
    "togethercomputer/RedPajama-INCITE-7B-Chat",
    format=RedpajamaFormat(),
)

MPT_7B_INSTRUCT = ModelConfig(
    "mosaicml/mpt-7b-instruct",
    format=DollyFormat(),
    model_kwargs={
        "init_device": "meta",
        # MPT not yet full supported by Transformers
        "trust_remote_code": True,
    },
)

MPT_7B_CHAT = ModelConfig(
    "mosaicml/mpt-7b-chat",
    format=ChatMLFormat(),
    model_kwargs={
        "init_device": "meta",
        # MPT not yet full supported by Transformers
        "trust_remote_code": True,
    },
)

MPT_30B_CHAT = ModelConfig(
    "mosaicml/mpt-30b-chat",
    format=ChatMLFormat(),
    model_kwargs={
        "init_device": "meta",
        # MPT not yet full supported by Transformers
        "trust_remote_code": True,
    },
)
