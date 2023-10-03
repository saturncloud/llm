from saturnfs import SaturnFS
import fsspec

fsspec.register_implementation("sfs", SaturnFS)

import os
import datetime
import pathlib
from typing import Tuple

from datasets import load_dataset, load_from_disk
import torch
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    prepare_model_for_int8_training,
)
from peft.utils import get_peft_model_state_dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    default_data_collator,
    LlamaForCausalLM,
    LlamaTokenizer,
)

from llm import settings
from llm.training.data import LazySupervisedFineTuning, process_pubmed_qa

# os.environ["WANDB_PROJECT"] = "llm-qa" # log to your project
# os.environ["WANDB_LOG_MODEL"] = "all" # log your models

## Create an experiment with your api key
# experiment = Experiment(
#     api_key=os.getenv('COMET_API_KEY'),
#     project_name="llm-fine-tuning",
#     workspace="llm",
#     auto_output_logging='native'
# )

# NAME = "qa-docs-vicuna"
# NAME = "qa-docs-llama"
NAME = "dummy-animals-llama2-concat"
BASE_MODEL = "meta-llama/Llama-2-7b-hf"
train_dataset_id = "sfs://internal/hugo/dummy-animals/train-fine-tuning"
eval_dataset_id = "sfs://internal/hugo/dummy-animals/eval-fine-tuning"


comet_ml.init(
    api_key=os.getenv("COMET_API_KEY"),
    project_name=NAME,
    workspace="llm",
)

# MODEL_MAX_LENGTH = 2048
MODEL_MAX_LENGTH = 4096
DEBUGGING = True
RESUME_FROM_CHECKPOINT = True

WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
GLOBAL_RANK = int(os.getenv("RANK", 0))


def has_checkpoint(dir: str) -> bool:
    if list(pathlib.Path(dir).glob("checkpoint-*")):
        return True
    return False


def load_base_model(lora_config: LoraConfig) -> Tuple[PeftModel, PreTrainedTokenizerBase]:
    tokenizer = LlamaTokenizer.from_pretrained(
        BASE_MODEL,
    )
    device_map = {"": LOCAL_RANK}
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model.train()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    # Load LoRa adapter
    model = get_peft_model(model, lora_config)

    if LOCAL_RANK == 0:
        model.print_trainable_parameters()

    return model, tokenizer


def tune():
    output_dir = os.path.join(settings.LOCAL_MODELS_DIR, NAME)
    model_dir_first = os.path.join(output_dir, "model1")
    model_dir_second = os.path.join(output_dir, "model2")

    train_dataset = load_from_disk(train_dataset_id)
    eval_dataset = load_from_disk(eval_dataset_id)
    max_eval_samples = 30
    eval_dataset = eval_dataset.select(range(0, max_eval_samples))
    training_args = TrainingArguments(
        output_dir,
        bf16=False,
        # logging strategies
        logging_strategy="steps",
        logging_steps=10,
        optim="adamw_torch_fused",
        save_strategy="no",
        report_to="comet_ml",
        learning_rate=1e-4,
        num_train_epochs=1,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=2,
        gradient_checkpointing=True,
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        inference_mode=False,
    )
    model, tokenizer = load_base_model(lora_config)

    print(training_args)
    trainer = Trainer(
        model,
        args=training_args,
        data_collator=default_data_collator,
        train_dataset=train_dataset,
    )
    # trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT and has_checkpoint(output_dir))
    trainer.train()
    model.save_pretrained(model_dir_first)
    state_dict = get_peft_model_state_dict(model)
    if GLOBAL_RANK == 0:
        model.save_pretrained(model_dir_second, state_dict=state_dict)


if __name__ == "__main__":
    tune()
