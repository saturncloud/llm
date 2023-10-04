"""
This script converts data to a format that is suitable for training. We expect inputs to be a HuggingFace dataset that has the following fields:
input: str
output: Optional[str]
contexts: List[str] # contexts you want the LLM to use in the output.
"""
from typing import Dict, List, Any
import copy

import click
from datasets import Dataset, Sequence, Value, Features
from ruamel.yaml import YAML
from transformers import PreTrainedTokenizer, AutoTokenizer
import torch

from llm.model_configs import ModelConfig
from llm.prompt import Message, Prompt
from llm.training.config import DataPrepConfig


class TrainingFormatConverter:
    def __init__(
        self,
        prompt: Prompt,
        tokenizer: PreTrainedTokenizer,
        chunksize: int,
        ignore_index: int = -100,
    ):
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.chunksize = chunksize
        self.ignore_index = ignore_index

    def __call__(self, batch: Dict[str, List]) -> Dict[str, List]:
        prompt = self.prompt
        tokenizer = self.tokenizer
        chunksize = self.chunksize
        ignore_index = self.ignore_index
        num_rows = len(batch["input"])
        output = {}
        for idx in range(num_rows):
            input_content = batch["input"][idx]
            response_content = batch["response"][idx]
            if "contexts" in batch:
                contexts = batch["contexts"][idx]
            else:
                contexts = None
            input_message = Message(input=input_content, response="", contexts=contexts)
            full_message = Message(
                input=input_content, response=response_content, contexts=contexts
            )
            input_prompt = prompt.render([input_message])
            full_text = prompt.render([full_message])
            prompt_length = len(tokenizer.encode(input_prompt))
            input_ids = tokenizer.encode(full_text)

            if tokenizer.bos_token_id is not None and input_ids[0] != tokenizer.bos_token_id:
                input_ids.insert(tokenizer.bos_token_id, 0)
            if tokenizer.eos_token_id is not None and input_ids[-1] != tokenizer.eos_token_id:
                input_ids.append(tokenizer.eos_token_id)

            if len(input_ids) >= chunksize:
                continue

            input_ids = torch.tensor(input_ids)
            labels = copy.deepcopy(input_ids)
            labels[:prompt_length] = ignore_index
            attention_mask = input_ids.ge(0).short()
            output.setdefault("input_ids", []).append(input_ids)
            output.setdefault("attention_mask", []).append(attention_mask)
            output.setdefault("labels", []).append(labels)

        return output


class Concatenator(object):
    fields = ["input_ids", "attention_mask", "labels"]

    def __init__(
        self,
        chunk_size: int,
        total_number_of_examples: int,
        unk_id: int = 0,
        ignore_index: int = -100,
    ):
        self.total_number_of_examples = total_number_of_examples
        self.chunk_size = chunk_size
        self.residual = {"input_ids": [], "attention_mask": [], "labels": []}
        self.to_process = None
        self.output = None
        self.pad_value = {"input_ids": unk_id, "labels": ignore_index, "attention_mask": 0}

    def add_new_output_entry(self):
        """
        adds a new row to self.output, which is empty
        """
        if self.output is None:
            self.output = {}
        for field in self.fields:
            self.output.setdefault(field, []).append([])

    def ensure_output_is_initialized(self):
        """
        makes sure self.output is valid, and has at least one row (even if it's empty)
        """
        if self.output is None:
            self.add_new_output_entry()
        input_ids = self.output["input_ids"]
        if len(input_ids) == 0:
            self.add_new_output_entry()

    def last_output_length(self):
        """
        returns the number of items in the last row of what we are about to output
        """
        self.ensure_output_is_initialized()
        return len(self.output["input_ids"][-1])

    def rows_to_process(self):
        """
        how many rows do we have left to process
        """
        return len(self.to_process["input_ids"])

    def next_input_length(self):
        """
        how long is the next input row we are planning to accumulate
        """
        return len(self.to_process["input_ids"][0])

    def accumulate(self):
        """
        accumulates the next pending input into the output
        removes that row from the input
        """
        self.ensure_output_is_initialized()
        for f in self.fields:
            self.output[f][-1].extend(self.to_process[f][0])
        for f in self.fields:
            self.to_process[f].pop(0)

    def pad_last_output_row(self):
        """
        finalizes the last output. pads it to max length, and
        initializes the next empty output for continued processing
        """
        to_pad = self.chunk_size - self.last_output_length()
        for f in self.fields:
            self.output[f][-1].extend(to_pad * [self.pad_value[f]])

    def move_last_output_row_to_residual(self):
        """
        We always save the last output row as a residual for the next batch, unless this is
        the final batch of the dataset. This method moves the last row to the residual
        """
        for f in self.fields:
            self.residual[f] = [self.output[f][-1]]
            self.output[f].pop(-1)

    def check_for_input_too_long(self):
        return self.next_input_length() > self.chunk_size

    def is_last_output_full(self):
        self.ensure_output_is_initialized()
        if self.last_output_length() > self.chunk_size:
            raise ValueError("output is too long. This should never happen")
        return self.last_output_length() == self.chunk_size

    def can_accumulate_without_exceeding_chunk_size(self):
        return (self.last_output_length() + self.next_input_length()) <= self.chunk_size

    @classmethod
    def concat(cls, batch1: Dict[str, List], batch2: Dict[str, List]) -> Dict[str, List]:
        output = {}
        for f in cls.fields:
            output[f] = batch1.get(f, []) + batch2.get(f, [])
        return output

    def prepare_data_to_process(self, batch: Dict[str, List]):
        """
        combines the residual with the new batch
        """
        self.to_process = self.concat(self.residual, batch)

    def __call__(self, batch: Dict[str, List], idx: List[int]):
        self.output = None
        self.ensure_output_is_initialized()
        self.prepare_data_to_process(batch)

        is_last_batch = max(idx) == self.total_number_of_examples - 1
        while self.rows_to_process() > 0:
            if self.check_for_input_too_long():
                raise ValueError("input is too long to be accumulated")

            if self.is_last_output_full():
                self.add_new_output_entry()

            if self.can_accumulate_without_exceeding_chunk_size():
                self.accumulate()
            else:
                self.pad_last_output_row()

        if is_last_batch:
            self.pad_last_output_row()
        else:
            self.move_last_output_row_to_residual()
        output = self.output
        self.output = None
        return output


def prepare_for_training(
    dataset: Dataset,
    prompt: Prompt,
    tokenizer: PreTrainedTokenizer,
    chunksize: int,
    ignore_index: int = -100,
    num_proc: int = 1,
) -> Dataset:
    converter = TrainingFormatConverter(
        prompt=prompt, tokenizer=tokenizer, chunksize=chunksize, ignore_index=ignore_index
    )
    dataset = dataset.map(
        converter,
        batched=True,
        remove_columns=dataset.features,
        num_proc=num_proc,
        features=Features(
            {
                "input_ids": Sequence(Value(dtype="int32")),
                "labels": Sequence(Value(dtype="int32")),
                "attention_mask": Sequence(Value(dtype="int16")),
            }
        ),
    )
    concatenator = Concatenator(
        chunk_size=chunksize,
        total_number_of_examples=len(dataset),
        unk_id=0,
        ignore_index=ignore_index,
    )
    dataset = dataset.map(
        concatenator,
        batched=True,
        with_indices=True,
        num_proc=1,
        features=Features(
            {
                "input_ids": Sequence(Value(dtype="int32")),
                "labels": Sequence(Value(dtype="int32")),
                "attention_mask": Sequence(Value(dtype="int16")),
            }
        ),
    )
    return dataset


@click.command()
@click.argument("config_path")
def run(config_path: str):
    with open(config_path) as f:
        config = YAML().load(f)
    _run(config)


def _run(config: Dict[str, Any]):
    dataprep_config = DataPrepConfig.from_config(**config)
    dataset = dataprep_config.source_config.load()
    tokenizer = AutoTokenizer.from_pretrained(dataprep_config.base_model)
    model_config = ModelConfig.from_registry(config["base_model"])
    prompt = dataprep_config.prompt_config.load(model_config.format)
    dataset = prepare_for_training(
        dataset,
        prompt,
        tokenizer,
        dataprep_config.chunksize,
        dataprep_config.ignore_index,
        dataprep_config.num_proc,
    )
    dataprep_config.dataset_writer_config.save(dataset)


if __name__ == "__main__":
    run()
