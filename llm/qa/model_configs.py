from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type
import torch
from fastchat.conversation import Conversation, SeparatorStyle
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from llm.qa import prompts


default_model_kwargs = {
    "load_in_8bit": True,
    "torch_dtype": torch.float16,
    "device_map": "auto",
}
default_tokenizer_kwargs = {
    "use_fast": True,
}
default_conversation_kwargs = {
    "roles": ("Question", "Answer"),
    "sep_style": SeparatorStyle.ADD_COLON_SINGLE,
    "sep": "\n",
    "stop_str": "Question:",
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
    name: str
    max_length: int = 512
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    merge_defaults: bool = True

    def __post_init__(self):
        if self.merge_defaults:
            self.model_kwargs = merge_dict(self.model_kwargs, default_model_kwargs)
            self.tokenizer_kwargs = merge_dict(self.tokenizer_kwargs, default_tokenizer_kwargs)

    def load(
        self,
        model_cls: Optional[Type[PreTrainedModel]] = None,
        tokenizer_cls: Optional[Type[PreTrainedTokenizerBase]] = None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        m_cls = model_cls or AutoModelForCausalLM
        t_cls = tokenizer_cls or AutoTokenizer
        model = m_cls.from_pretrained(self.name, **self.model_kwargs)
        tokenizer = t_cls.from_pretrained(self.name, **self.tokenizer_kwargs)
        return model, tokenizer


@dataclass
class ChatModelConfig(ModelConfig):
    max_length: int = 2048
    conversation_kwargs: Dict[str, Any] = field(default_factory=dict)
    default_prompt: prompts.ContextPrompt = prompts.ZERO_SHOT

    def __post_init__(self):
        if self.merge_defaults:
            super().__post_init__()
            self.conversation_kwargs = merge_dict(self.conversation_kwargs, default_conversation_kwargs)

    def new_conversation(self) -> Conversation:
        kwargs = {
            "name": self.name,
            "system": "",
            "messages": [],
            "offset": 0,
            **self.conversation_kwargs,
        }
        return Conversation(**kwargs)


VICUNA = ChatModelConfig(
    "lmsys/vicuna-7b-v1.3",
    tokenizer_kwargs={
        # Llama fast tokenizer is not good
        "use_fast": False
    },
    default_prompt=prompts.FEW_SHOT,
)

REDPAJAMA_INSTRUCT = ChatModelConfig(
    "togethercomputer/RedPajama-INCITE-7B-Instruct",
    default_prompt=prompts.FEW_SHOT,
    conversation_kwargs={
        "roles": ("<human>", "<bot>"),
        "stop_str": "<human>:",
    },
)

REDPAJAMA_CHAT = ChatModelConfig(
    "togethercomputer/RedPajama-INCITE-7B-Chat",
    conversation_kwargs={
        "roles": ("<human>", "<bot>"),
        "stop_str": "<human>:",
    }
)

MPT_INSTRUCT = ChatModelConfig(
    "mosaicml/mpt-7b-instruct",
    model_kwargs={
        "init_device": "cuda:0",
        # MPT not yet full supported by Transformers
        "trust_remote_code": True,
        "revision": "e7119f37956c1a3865da33e25ef5ce9159ff2c16",
    },
    default_prompt=prompts.INSTRUCTION_FEW_SHOT,
)

MPT_CHAT = ChatModelConfig(
    "mosaicml/mpt-7b-chat",
    model_kwargs={
        "init_device": "cuda:0",
        # MPT not yet full supported by Transformers
        "trust_remote_code": True,
        "revision": "c8d4750ac8421303665d6ecc253950c69b56d324",
    },
    default_prompt=prompts.ZERO_SHOT,
)
