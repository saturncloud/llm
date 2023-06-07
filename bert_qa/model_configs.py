from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type
import torch
from fastchat.conversation import Conversation, SeparatorStyle
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from bert_qa import prompts


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
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    merge_defaults: bool = True

    def __post_init__(self):
        if not self.merge_defaults:
            return

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
class QAModelConfig(ModelConfig):
    conversation_kwargs: Dict[str, Any] = field(default_factory=dict)
    default_prompt_kwargs: Dict[str, Any] = field(default_factory=dict)
    default_prompt: prompts.QAPrompt = prompts.ZERO_SHOT
    context_label: str = "Context"

    def __post_init__(self):
        if not self.merge_defaults:
            return
        super().__post_init__()

        self.conversation_kwargs = merge_dict(self.conversation_kwargs, default_conversation_kwargs)

    def create_conversation(self) -> Conversation:
        kwargs = {
            "name": self.name,
            "system": "",
            "messages": [],
            "offset": 0,
            **self.conversation_kwargs,
        }
        return Conversation(**kwargs)

    def render_prompt(
        self,
        prompt: Optional[prompts.QAPrompt] = None,
        merge_defaults: bool = True,
        **prompt_kwargs,
    ) -> str:
        if not prompt:
            prompt = self.default_prompt
        if prompt_kwargs and merge_defaults:
            prompt_kwargs = merge_dict(prompt_kwargs, self.default_prompt_kwargs)
        elif not prompt_kwargs:
            prompt_kwargs = self.default_prompt_kwargs
        return self.default_prompt.render(**prompt_kwargs)


VICUNA = QAModelConfig(
    "/home/jovyan/workspace/models/vicuna-7b",
    tokenizer_kwargs={
        # Llama fast tokenizer is not good
        "use_fast": False
    },
    conversation_kwargs={
        "stop_str": "Question:"
    },
)

REDPAJAMA = QAModelConfig(
    "togethercomputer/RedPajama-INCITE-7B-Instruct",
    default_prompt=prompts.FEW_SHOT,
)

MPT_INSTRUCT = QAModelConfig(
    "mosaicml/mpt-7b-instruct",
    model_kwargs={
        "trust_remote_code": True,
        "config": AutoConfig.from_pretrained(
            "mosaicml/mpt-7b-instruct",
            trust_remote_code=True,
            init_device="cuda:0",
            max_seq_length=2048,
        )
    },
    default_prompt=prompts.INSTRUCTION_FEW_SHOT,
)

MPT_CHAT = QAModelConfig(
    "mosaicml/mpt-7b-chat",
    model_kwargs={
        "trust_remote_code": True,
        "config": AutoConfig.from_pretrained(
            "mosaicml/mpt-7b-chat",
            trust_remote_code=True,
            init_device="cuda:0",
            max_seq_length=2048,
        )
    },
    default_prompt=prompts.ZERO_SHOT,
)
