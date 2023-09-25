from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase

from llm.utils.data import merge_dict
from llm.prompt import Llama2Format, Conversation, RedpajamaFormat, TogetherLlama2Format, VicunaFormat

default_model_kwargs = {
    "load_in_8bit": True,
    "torch_dtype": torch.float16,
    "device_map": "auto",
}
default_tokenizer_kwargs = {
    "use_fast": True,
    # https://github.com/huggingface/transformers/pull/24565
    "legacy": False,
}
default_conversation_kwargs = {}

_registry: Dict[str, ModelConfig] = {}


@dataclass
class ModelConfig:
    """
    Stores model and tokenizer configuration for
    pretrained huggingface models.
    """
    model_id: str
    max_length: int = 512
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

    def load(self, device_map: Optional[Union[str, Dict]] = None) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        return self.load_model(device_map), self.load_tokenizer()

    def load_model(self, device_map: Optional[Union[str, Dict]] = None) -> PreTrainedModel:
        model_cls = self.model_cls or AutoModelForCausalLM
        if device_map is not None:
            model_kwargs = {
                **self.model_kwargs,
                "device_map": device_map,
            }
        else:
            model_kwargs = self.model_kwargs
        model = model_cls.from_pretrained(self.model_id, **model_kwargs)
        if self.peft_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.peft_adapter, **self.peft_kwargs)
        return model

    def load_tokenizer(self) -> PreTrainedTokenizerBase:
        tokenizer_cls = self.tokenizer_cls or AutoTokenizer
        return tokenizer_cls.from_pretrained(self.model_id, **self.tokenizer_kwargs)


@dataclass
class ChatModelConfig(ModelConfig):
    """
    Stores model, tokenizer, and conversation configuration for
    pretrained huggingface models.
    """
    max_length: int = 2048
    conversation_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.merge_defaults:
            super().__post_init__()
            self.conversation_kwargs = merge_dict(self.conversation_kwargs, default_conversation_kwargs)

    def new_conversation(self) -> Conversation:
        return Conversation(**self.conversation_kwargs)


def trim_model_path(model_id: str) -> str:
    """
    Strip trailing / from path-based model IDs for consistency.
    """
    if os.path.isdir(model_id) and model_id.endswith("/"):
        return model_id.rstrip("/")
    return model_id


VICUNA_7B = ChatModelConfig(
    "lmsys/vicuna-7b-v1.5",
    tokenizer_kwargs={
        # Llama fast tokenizer is not good
        "use_fast": False,
    },
    conversation_kwargs={
        "format": VicunaFormat(),
    }
)

VICUNA_13B = ChatModelConfig(
    "lmsys/vicuna-13b-v1.5",
    max_length=4096,
    tokenizer_kwargs={
        "use_fast": False
    },
    conversation_kwargs={
        "format": VicunaFormat(),
    }
)

VICUNA_33B = ChatModelConfig(
    "lmsys/vicuna-33b-v1.5",
    tokenizer_kwargs={
        "use_fast": False
    },
    conversation_kwargs={
        "format": VicunaFormat(),
    }
)

LLAMA2_7B = ChatModelConfig(
    "meta-llama/Llama-2-7b-chat-hf",
    max_length=4096,
    conversation_kwargs={
        "format": Llama2Format(),
    },
)

LLAMA2_13B = ChatModelConfig(
    "meta-llama/Llama-2-13b-chat-hf",
    max_length=4096,
    model_kwargs={
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        ),
    },
    conversation_kwargs={
        "format": Llama2Format(),
    },
)

LLAMA2_7B_32K = ChatModelConfig(
    "togethercomputer/LLaMA-2-7B-32K",
    max_length=32768,
    conversation_kwargs={
        "format": TogetherLlama2Format(),
    },
)

LLAMA2_7B_32K_INSTRUCT = ChatModelConfig(
    "togethercomputer/Llama-2-7B-32K-Instruct",
    max_length=32768,
    conversation_kwargs={
        "format": TogetherLlama2Format(),
    },
)

REDPAJAMA_INSTRUCT = ChatModelConfig(
    "togethercomputer/RedPajama-INCITE-7B-Instruct",
    conversation_kwargs={
        "format": RedpajamaFormat(),
    },
)

REDPAJAMA_CHAT = ChatModelConfig(
    "togethercomputer/RedPajama-INCITE-7B-Chat",
    conversation_kwargs={
        "format": RedpajamaFormat(),
    }
)

MPT_INSTRUCT = ChatModelConfig(
    "mosaicml/mpt-7b-instruct",
    model_kwargs={
        "init_device": "cuda:0",
        # MPT not yet full supported by Transformers
        "trust_remote_code": True,
        "revision": "1fc4634127ec64a45716003578b9cfae23265849",
    },
)

MPT_CHAT = ChatModelConfig(
    "mosaicml/mpt-7b-chat",
    model_kwargs={
        "init_device": "cuda:0",
        # MPT not yet full supported by Transformers
        "trust_remote_code": True,
        "revision": "c53dee01e05098f81cac11145f9bf45feedc5b2f",
    },
)
