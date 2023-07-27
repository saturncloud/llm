from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type, Union
import torch
from langchain.memory.buffer_window import ConversationBufferWindowMemory
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from llm import settings
from llm.qa import prompts


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
default_conversation_kwargs = {
    "human_prefix": "Question",
    "ai_prefix": "Answer",
}


def merge_dict(a: Dict, b: Dict) -> Dict:
    new = deepcopy(b)
    for k, v in a.items():
        if isinstance(v, dict):
            new[k] = merge_dict(v, new.get(k, {}))
        else:
            new[k] = v
    return new


@dataclass
class ModelConfig:
    """
    Stores model and tokenizer configuration for
    pretrained huggingface models.
    """
    name: str
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

    def load(self, device_map: Optional[Union[str, Dict]] = None) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        model_cls = self.model_cls or AutoModelForCausalLM
        tokenizer_cls = self.tokenizer_cls or AutoTokenizer
        if device_map is not None:
            model_kwargs = {
                **self.model_kwargs,
                "device_map": device_map,
            }
        else:
            model_kwargs = self.model_kwargs
        model = model_cls.from_pretrained(self.name, **model_kwargs)
        if self.peft_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.peft_adapter, **self.peft_kwargs)
        tokenizer = tokenizer_cls.from_pretrained(self.name, **self.tokenizer_kwargs)
        return model, tokenizer


@dataclass
class ChatModelConfig(ModelConfig):
    """
    Stores model, tokenizer, and conversation configuration for
    pretrained huggingface models.
    """
    max_length: int = 2048
    conversation_kwargs: Dict[str, Any] = field(default_factory=dict)
    default_prompt: prompts.ContextPrompt = prompts.ZERO_SHOT

    def __post_init__(self):
        if self.merge_defaults:
            super().__post_init__()
            self.conversation_kwargs = merge_dict(self.conversation_kwargs, default_conversation_kwargs)

    def new_conversation(self) -> ConversationBufferWindowMemory:
        return ConversationBufferWindowMemory(**self.conversation_kwargs)


VICUNA_7B = ChatModelConfig(
    "lmsys/vicuna-7b-v1.3",
    tokenizer_kwargs={
        # Llama fast tokenizer is not good
        "use_fast": False,
    },
    default_prompt=prompts.FEW_SHOT,
)

VICUNA_13B = ChatModelConfig(
    "lmsys/vicuna-13b-v1.3",
    tokenizer_kwargs={
        "use_fast": False
    },
    default_prompt=prompts.FEW_SHOT,
)

VICUNA_33B = ChatModelConfig(
    "lmsys/vicuna-33b-v1.3",
    tokenizer_kwargs={
        "use_fast": False
    },
    default_prompt=prompts.FEW_SHOT,
)

MEDCUNA_7B = ChatModelConfig(
    "lmsys/vicuna-7b-v1.3",
    tokenizer_kwargs={
        "use_fast": False
    },
    default_prompt=prompts.ZERO_SHOT,
    peft_adapter=os.path.join(settings.LOCAL_MODELS_DIR, "/medcuna-7b"),
)

LLAMA2_7B = ChatModelConfig(
    "meta-llama/Llama-2-7b-chat-hf",
    max_length=4096,
    default_prompt=prompts.FEW_SHOT,
    conversation_kwargs={
        "roles": ("user", "assistant"),
        "stop_str": "user",
    }
)

REDPAJAMA_INSTRUCT = ChatModelConfig(
    "togethercomputer/RedPajama-INCITE-7B-Instruct",
    default_prompt=prompts.FEW_SHOT,
    conversation_kwargs={
        "human_prefix": "<human>",
        "ai_prefix": "<bot>",
    },
)

REDPAJAMA_CHAT = ChatModelConfig(
    "togethercomputer/RedPajama-INCITE-7B-Chat",
    conversation_kwargs={
        "human_prefix": "<human>",
        "ai_prefix": "<bot>",
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
    default_prompt=prompts.INSTRUCTION_FEW_SHOT,
)

MPT_CHAT = ChatModelConfig(
    "mosaicml/mpt-7b-chat",
    model_kwargs={
        "init_device": "cuda:0",
        # MPT not yet full supported by Transformers
        "trust_remote_code": True,
        "revision": "c53dee01e05098f81cac11145f9bf45feedc5b2f",
    },
    default_prompt=prompts.ZERO_SHOT,
)
