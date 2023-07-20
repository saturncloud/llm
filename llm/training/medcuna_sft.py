import os
import logging
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
import deepspeed
import torch
from trl import SFTTrainer
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from peft.utils import get_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, PreTrainedTokenizerBase, Trainer, TrainingArguments

from llm.qa.prompts import ZERO_SHOT


BASE_MODEL = "lmsys/vicuna-7b-v1.3"
MODEL_MAX_LENGTH = 2048
TRAIN_DATASET = {
    "path": "pubmed_qa",
    "name": "pqa_unlabeled",
    "split": "train[:20]",
}
EVAL_DATASET = {
    "path": "pubmed_qa",
    "name": "pqa_unlabeled",
    "split": "train[-6000:]",
}

WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", "models/")
TRAINING_DIR = os.path.dirname(__file__)


def prompt_formatting_func(batch: Dict[str, List[Any]]):
    prompts = []
    for question, context, long_answer in zip(batch["question"], batch["context"], batch["long_answer"]):
        prompt = ZERO_SHOT.render(context["contexts"], question=question)
        prompt += " " + long_answer
        prompts.append(prompt)
    return prompts


def load_base_model(lora_config: LoraConfig, gradient_checkpointing: bool = True) -> Tuple[PeftModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        model_max_length=MODEL_MAX_LENGTH,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Load model to GPU specified by deepspeed (if applicable), otherwise use auto
    device_map = (
        {"": LOCAL_RANK} if WORLD_SIZE != 1 else "auto"
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)

    if torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    model = get_peft_model(model, lora_config)
    model.config.use_cache=False

    # if gradient_checkpointing:
    #     model.enable_input_require_grads()

    if LOCAL_RANK == 0:
        model.print_trainable_parameters()

    return model, tokenizer


def tune():
    output_dir = os.path.join(MODEL_OUTPUT_DIR, "medcuna-7b/")
    # TODO: Need to wrap in something like LazySupervisedDataset
    dataset = load_dataset("pubmed_qa", "pqa_unlabeled", split="train[:10%]")

    training_args = TrainingArguments(
        output_dir,
        do_train=True,
        bf16=True,
        optim="adamw_torch_fused",
        auto_find_batch_size=True,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        logging_steps=1,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=20,
        learning_rate=2e-5,
        warmup_ratio=0.001,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        deepspeed=os.path.join(TRAINING_DIR, "deepspeed_configs/bf16.json"),
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model, tokenizer = load_base_model(lora_config, gradient_checkpointing=training_args.gradient_checkpointing)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # trainer = Trainer(
    #     model,
    #     args=training_args,
    #     data_collator=data_collator,
    #     train_dataset=dataset,
    #     tokenizer=tokenizer,
    #     # max_seq_length=MODEL_MAX_LENGTH,
    # )
    trainer = SFTTrainer(
        model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=prompt_formatting_func,
        max_seq_length=MODEL_MAX_LENGTH,
        packing=False,
    )
    trainer.train()

    state_dict = get_peft_model_state_dict(model)
    if LOCAL_RANK == 0:
        model.save_pretrained(output_dir, state_dict=state_dict)


if __name__ == "__main__":
    tune()
