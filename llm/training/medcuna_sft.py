import os
import datetime
import pathlib
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from peft.utils import get_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, PreTrainedTokenizerBase, Trainer, TrainingArguments

from llm.qa.prompts import ZERO_SHOT


NAME = "medcuna-13b"
BASE_MODEL = "lmsys/vicuna-13b-v1.3"
MODEL_MAX_LENGTH = 2048
TRAIN_DATASET = {
    "path": "pubmed_qa",
    "name": "pqa_unlabeled",
    "split": "train[:6000]",
}
EVAL_DATASET = {
    "path": "pubmed_qa",
    "name": "pqa_unlabeled",
    "split": "train[-20:]",
}
IGNORE_TOKEN_ID = -100
DEBUGGING = True

WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
GLOBAL_RANK = int(os.getenv("RANK", 0))
TRAINING_DIR = os.path.dirname(__file__)
LOCAL_MODELS_DIR = os.getenv("LOCAL_MODELS_DIR", "models/")
SATURNFS_MODELS_DIR = os.getenv("SATURNFS_MODELS_DIR", None)


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

    # Load model to GPU specified by deepspeed (if applicable), otherwise use auto
    device_map = (
        {"": LOCAL_RANK} if WORLD_SIZE != 1 else "auto"
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

    model = get_peft_model(model, lora_config)
    model.config.use_cache=False

    if LOCAL_RANK == 0:
        model.print_trainable_parameters()

    return model, tokenizer


def process_data(
    qa_example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> Dict:
    roles = ["Question", "Answer"]
    context: Dict[str, Any] = qa_example["context"]
    question: str = qa_example["question"]
    answer: str = qa_example["long_answer"]

    prompt = ZERO_SHOT.render(
        context["contexts"], roles=roles, question=question, answer=answer
    )

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
    sep = f"\n{roles[1]}: "
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


class LazySupervisedDataset(torch.utils.data.Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: PreTrainedTokenizerBase, cache_data: bool = False):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = process_data(self.raw_data[i], self.tokenizer)
        self.cached_data_dict[i] = ret
        return ret


def tune():
    output_dir = os.path.join(LOCAL_MODELS_DIR, NAME)
    dataset = load_dataset(**TRAIN_DATASET)

    training_args = TrainingArguments(
        output_dir,
        do_train=True,
        bf16=True,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        auto_find_batch_size=True,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        save_strategy="epoch",
        learning_rate=2e-5,
        warmup_ratio=0.001,
        logging_steps=1,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
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
    supervised_dataset = LazySupervisedDataset(dataset, tokenizer)

    trainer = Trainer(
        model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=supervised_dataset,
        tokenizer=tokenizer,
    )
    trainer.train(resume_from_checkpoint=has_checkpoint(output_dir))

    state_dict = get_peft_model_state_dict(model)
    if GLOBAL_RANK == 0:
        model.save_pretrained(output_dir, state_dict=state_dict)
        if SATURNFS_MODELS_DIR:
            from saturnfs import SaturnFS
            timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H.%M.%S")
            remote_dir = os.path.join(SATURNFS_MODELS_DIR, f"{NAME}-{timestamp}")
            sfs = SaturnFS()
            sfs.put(output_dir, remote_dir, recursive=True)


if __name__ == "__main__":
    tune()
