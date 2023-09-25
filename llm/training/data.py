from typing import Any, Callable, Dict

import torch
from transformers import PreTrainedTokenizerBase
from datasets import Dataset

from llm.prompt import Message, PromptFormat, VicunaFormat
from llm.qa.prompts import ZeroShotQA

IGNORE_TOKEN_ID = -100


class LazySupervisedFineTuning(torch.utils.data.Dataset):
    """
    Dataset for supervised fine-tuning.
    """

    def __init__(
        self,
        raw_data: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        process_data: Callable[[Dict[str, Any], PreTrainedTokenizerBase], Dict[str, Any]],
    ):
        super().__init__()
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.process_data = process_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = self.process_data(self.raw_data[i], self.tokenizer)
        self.cached_data_dict[i] = ret
        return ret


def process_pubmed_qa(
    qa_example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    format: PromptFormat = VicunaFormat(),
) -> Dict:
    """
    Apply a ZeroShotQA template on a PubmedQA example,
    tokenizes and formats the fields for supervised fine-tuning.
    """
    format = VicunaFormat()
    context: Dict[str, Any] = qa_example["context"]
    question: str = qa_example["question"]
    answer: str = qa_example["long_answer"]
    message = Message(question, answer, contexts=context["contexts"])

    prompt = ZeroShotQA().render([message], format=format)

    # Tokenize conversations
    input_ids: torch.Tensor = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids[0]
    target = input_ids.clone()

    # Mask targets. Only want to train on responses to the questions, not on how
    # to generate questions/contexts.
    sep = format.user.prefix
    split = prompt.split(sep)
    assert len(split) == 2
    split[0] += sep

    instruction_len = len(tokenizer(split[0], add_special_tokens=False).input_ids)
    target[:instruction_len] = IGNORE_TOKEN_ID

    return dict(
        input_ids=input_ids,
        labels=target,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
