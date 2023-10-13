import os
from typing import Dict, Any
from os.path import join
import pathlib

import torch

# import experiment tracking imports before ML toolkits
import llm.experiment_tracking_imports  # noqa
import click
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    Trainer,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training, get_peft_model
from ruamel.yaml import YAML

from llm.training.config import FineTuneConfig, CopyToSourcesCallback


WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
GLOBAL_RANK = int(os.getenv("RANK", 0))


def has_checkpoint(dir: str) -> bool:
    if list(pathlib.Path(dir).glob("checkpoint-*")):
        return True
    return False


@click.command()
@click.argument("config_path")
def run(config_path: str):
    with open(config_path) as f:
        config = YAML().load(f)
    _run(config)


def _run(config: Dict[str, Any]):
    finetune_config = FineTuneConfig.from_config(**config)

    tokenizer = AutoTokenizer.from_pretrained(finetune_config.base_model)
    device_map = {"": LOCAL_RANK}
    model = AutoModelForCausalLM.from_pretrained(
        finetune_config.base_model,
        load_in_8bit=finetune_config.load_in_8bit,
        load_in_4bit=finetune_config.load_in_4bit,
        torch_dtype=finetune_config.torch_dtype,
        device_map=device_map,
        quantization_config=finetune_config.quantization_config,
    )
    model.train()
    if finetune_config.is_quantized:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=finetune_config.training_arguments.gradient_checkpointing
        )
    model = get_peft_model(model, finetune_config.lora_config)

    if LOCAL_RANK == 0:
        model.print_trainable_parameters()
    train_dataset = finetune_config.train_dataset_config.load()
    eval_dataset = finetune_config.eval_dataset_config.load()
    copy_callback: CopyToSourcesCallback = finetune_config.copy_callback()
    trainer = Trainer(
        model,
        args=finetune_config.training_arguments,
        data_collator=default_data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[],
    )
    trainer.train(
        resume_from_checkpoint=finetune_config.resume_from_checkpoint
        and has_checkpoint(finetune_config.local_output)
    )
    model.save_pretrained(join(finetune_config.local_output, "final_output"))
    copy_callback.rsync()


if __name__ == "__main__":
    run()
