from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type, Union
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    __version__ as TRANSFORMERS_VERSION,
)
from transformers.modeling_utils import is_peft_available
from transformers.utils.quantization_config import QuantizationConfigMixin

from llm.prompt import (
    ChatMLFormat,
    DollyFormat,
    Llama2Format,
    PromptFormat,
    RedpajamaFormat,
    TogetherLlama2Format,
    VicunaFormat,
)
from llm.utils.logs import get_logger

logger = get_logger()
_registry: Dict[str, Type[ModelConfig]] = {}

QuantizationConfig = Union[str, Dict[str, Any], QuantizationConfigMixin]


def bnb_quantization(mode: str = "4bit") -> BitsAndBytesConfig:
    """
    Create a valid BitsAndBytes quantization for the version of transformers that is installed.
    """
    if mode == "4bit":
        if TRANSFORMERS_VERSION >= "4.30.0":
            # 4-bit supported after this version
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        logger.warning(
            f"Transformers version '{TRANSFORMERS_VERSION}' "
            "does not support 4-bit quantization. Falling back on 8-bit."
        )
    elif mode != "8bit":
        raise Exception(f"Unrecognized quantization mode '{mode}'. Supports '8bit' or '4bit'")
    return BitsAndBytesConfig(
        load_in_8bit=True,
    )


@dataclass
class ModelConfig:
    """
    Stores model and tokenizer configuration for
    pretrained huggingface models.

    PEFT models may be loaded by their adapter model_id
    assuming they have been properly created with adapter_config.json
    """

    model_id: str = ""
    max_length: int = 2048
    format: PromptFormat = field(default_factory=PromptFormat)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_cls: Optional[Type[PreTrainedModel]] = None
    tokenizer_cls: Optional[Type[PreTrainedTokenizerBase]] = None
    peft_base_id: Optional[str] = None
    peft_kwargs: Dict[str, Any] = field(default_factory=dict)
    default_lora_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.model_id = trim_model_path(self.model_id)

    def __init_subclass__(cls) -> None:
        if cls.model_id:
            cls.register(cls.model_id)

    @classmethod
    def register(cls, *model_ids: str):
        """
        Register additional model_ids for the class
        """
        for name in model_ids:
            _registry[trim_model_path(name)] = cls

    @classmethod
    def from_registry(cls, model_id: str, **kwargs) -> ModelConfig:
        """
        Load model config from the registry
        """
        model_id = trim_model_path(model_id)
        config_cls = cls
        if model_id in _registry:
            config_cls = _registry[model_id]
        else:
            peft_base_id = fetch_peft_base(model_id)
            if peft_base_id and peft_base_id in _registry:
                config_cls = _registry[peft_base_id]
            else:
                logger.warn(
                    f'ModelConfig "{model_id}" not found in registry. Using generic configuration.'
                )
        if not (cls == config_cls or issubclass(config_cls, cls)):
            logger.warn(f'Registry entry for "{model_id}" {config_cls} is not a subclass of {cls}')

        return config_cls(model_id=model_id, **kwargs)

    def load(
        self,
        device_map: Optional[Union[str, Dict]] = None,
        quantization_config: Optional[QuantizationConfig] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        tokenizer_kwargs = tokenizer_kwargs or {}
        if device_map is not None:
            kwargs["device_map"] = device_map
        model = self.load_model(quantization_config=quantization_config, **kwargs)
        tokenizer = self.load_tokenizer(**tokenizer_kwargs)
        return model, tokenizer

    def load_model(self, quantization_config: Optional[QuantizationConfig] = None, **kwargs) -> PreTrainedModel:
        model_cls = self.model_cls or AutoModelForCausalLM
        model_kwargs = {
            **self.model_kwargs,
            **kwargs,
        }
        if quantization_config:
            if isinstance(quantization_config, str):
                quantization_config = bnb_quantization(quantization_config)
            model_kwargs["quantization_config"] = quantization_config
        return model_cls.from_pretrained(self.model_id, **model_kwargs)

    def load_tokenizer(self, **kwargs) -> PreTrainedTokenizerBase:
        tokenizer_cls = self.tokenizer_cls or AutoTokenizer
        tokenizer_kwargs = {
            **self.tokenizer_kwargs,
            **kwargs,
        }
        try:
            return tokenizer_cls.from_pretrained(self.model_id, **tokenizer_kwargs)
        except Exception as e:
            # Check if model_id is a PEFT adapter
            peft_base_id = fetch_peft_base(self.model_id)
            if peft_base_id:
                # Load base model's tokenizer
                return tokenizer_cls.from_pretrained(peft_base_id, **tokenizer_kwargs)
            raise e


def trim_model_path(model_id: str) -> str:
    """
    Strip trailing / from path-based model IDs for consistency.
    """
    if os.path.isdir(model_id) and model_id.endswith("/"):
        return model_id.rstrip("/")
    return model_id


def llama_lora_config() -> Dict[str, Any]:
    return dict(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        inference_mode=False,
    )


def fetch_peft_base(model_id: str) -> Optional[str]:
    if is_peft_available():
        from peft import PeftConfig

        try:
            config = PeftConfig.from_pretrained(model_id)
            return config.base_model_name_or_path
        except Exception:
            pass
    return None


@dataclass
class LlamaBaseConfig(ModelConfig):
    default_lora_config: Dict = field(default_factory=llama_lora_config)


@dataclass
class VicunaConfig(LlamaBaseConfig):
    model_id: str = "lmsys/vicuna-7b-v1.5"
    max_length: int = 4096
    format: PromptFormat = field(default_factory=VicunaFormat)


VicunaConfig.register(
    "lmsys/vicuna-7b-v1.5",
    "lmsys/vicuna-13b-v1.5",
    "lmsys/vicuna-33b-v1.5",
)


@dataclass
class Llama2Config(LlamaBaseConfig):
    model_id: str = "meta-llama/Llama-2-7b-hf"
    max_length: int = 4096


Llama2Config.register(
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
)


@dataclass
class Llama2ChatConfig(LlamaBaseConfig):
    model_id: str = "meta-llama/Llama-2-7b-chat-hf"
    max_length: int = 4096
    format: PromptFormat = field(default_factory=Llama2Format)
    tokenizer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            # Llama2 does not have a pad token. For batched inference, using the unk_token
            # performs better than the standard recommendation of using eos_token
            "pad_token": "<unk>",
        }
    )


Llama2ChatConfig.register(
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
)


@dataclass
class TogetherLlama2Config(ModelConfig):
    model_id: str = "togethercomputer/LLaMA-2-7B-32K"
    max_length: int = 32768


@dataclass
class TogetherLlama2InstructConfig(ModelConfig):
    model_id: str = "togethercomputer/LLaMA-2-7B-32K-Instruct"
    max_length: int = 32768
    format: PromptFormat = field(default_factory=TogetherLlama2Format)


@dataclass
class RedpajamaInstructConfig(ModelConfig):
    model_id: str = "togethercomputer/RedPajama-INCITE-7B-Instruct"
    format: PromptFormat = field(default_factory=RedpajamaFormat)


@dataclass
class RedpajamaChatConfig(ModelConfig):
    model_id: str = "togethercomputer/RedPajama-INCITE-7B-Chat"
    format: PromptFormat = field(default_factory=RedpajamaFormat)


@dataclass
class MPTInstructConfig(ModelConfig):
    model_id: str = "mosaicml/mpt-7b-instruct"
    format: PromptFormat = field(default_factory=DollyFormat)
    model_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "init_device": "meta",
            # MPT not yet full supported by Transformers
            "trust_remote_code": True,
        }
    )


MPTInstructConfig.register(
    "mosaicml/mpt-7b-instruct",
    "mosaicml/mpt-30b-instruct",
)


@dataclass
class MPTChatConfig(ModelConfig):
    model_id: str = "mosaicml/mpt-7b-chat"
    format: PromptFormat = field(default_factory=ChatMLFormat)
    model_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "init_device": "meta",
            # MPT not yet full supported by Transformers
            "trust_remote_code": True,
        }
    )


MPTChatConfig.register(
    "mosaicml/mpt-7b-chat",
    "mosaicml/mpt-30b-chat",
)
