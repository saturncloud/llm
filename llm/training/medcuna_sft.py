"""
Fine-tune Vicuna on PubmedQA dataset
"""

import os
import datetime
import pathlib
from typing import Tuple

from datasets import load_dataset
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from peft.utils import get_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, PreTrainedTokenizerBase, Trainer, TrainingArguments

from llm import settings
from llm.model_configs import ChatModelConfig
from llm.training.data import LazySupervisedFineTuning, process_pubmed_qa


NAME = "medcuna-7b"
BASE_MODEL = "lmsys/vicuna-7b-v1.3"
MODEL_MAX_LENGTH = 2048
TRAIN_DATASET = {
    "path": "pubmed_qa",
    "name": "pqa_unlabeled",
    "split": "train[:-100]",
}
EVAL_DATASET = {
    "path": "pubmed_qa",
    "name": "pqa_unlabeled",
    "split": "train[-100:]",
}
DEBUGGING = True
RESUME_FROM_CHECKPOINT = True

WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
GLOBAL_RANK = int(os.getenv("RANK", 0))

MEDCUNA_7B = ChatModelConfig(
    BASE_MODEL,
    tokenizer_kwargs={
        "use_fast": False
    },
    peft_adapter=os.path.join(settings.LOCAL_MODELS_DIR, NAME),
)


def has_checkpoint(dir: str) -> bool:
    if list(pathlib.Path(dir).glob("checkpoint-*")):
        return True
    return False


def load_base_model(lora_config: LoraConfig, gradient_checkpointing: bool = True) -> Tuple[PeftModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        model_max_length=MODEL_MAX_LENGTH,
        padding_side="right",
        use_fast=False,
        legacy=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Load model to GPU specified by LOCAL_RANK
    device_map = (
        {"": LOCAL_RANK}
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map=device_map,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)

    if torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    # Load LoRa adapter
    model = get_peft_model(model, lora_config)
    model.config.use_cache=False

    if LOCAL_RANK == 0:
        model.print_trainable_parameters()

    return model, tokenizer


def tune():
    output_dir = os.path.join(settings.LOCAL_MODELS_DIR, NAME)
    train_dataset = load_dataset(**TRAIN_DATASET)
    eval_dataset = load_dataset(**EVAL_DATASET)

    training_args = TrainingArguments(
        output_dir,
        do_train=True,
        do_eval=True,
        bf16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        auto_find_batch_size=True,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        eval_accumulation_steps=16,
        learning_rate=2e-5,
        warmup_ratio=0.001,
        logging_steps=1,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        report_to="none",
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
    lazy_training = LazySupervisedFineTuning(train_dataset, tokenizer, process_data=process_pubmed_qa)
    lazy_eval = LazySupervisedFineTuning(eval_dataset, tokenizer, process_data=process_pubmed_qa)

    trainer = Trainer(
        model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=lazy_training,
        eval_dataset=lazy_eval,
        tokenizer=tokenizer,
    )
    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT and has_checkpoint(output_dir))

    state_dict = get_peft_model_state_dict(model)
    if GLOBAL_RANK == 0:
        model.save_pretrained(output_dir, state_dict=state_dict)
        if settings.SATURNFS_MODELS_DIR:
            from saturnfs import SaturnFS
            timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H.%M.%S")
            remote_dir = os.path.join(settings.SATURNFS_MODELS_DIR, f"{NAME}-{timestamp}")
            sfs = SaturnFS()
            sfs.put(output_dir, remote_dir, recursive=True)


if __name__ == "__main__":
    tune()
