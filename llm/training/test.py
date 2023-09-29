import os

from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from peft import prepare_model_for_kbit_training, get_peft_model

from llm.training.config import FineTuneConfig
from llm.model_configs import ModelConfig

WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
GLOBAL_RANK = int(os.getenv("RANK", 0))

print(bb)

bb.foo()


def run(config: Dict[str, Any]):
    finetune_config = FineTuneConfig.from_config(**config)

    tokenizer = AutoTokenizer.from_pretrained(
        finetune_config.base_model
    )
    device_map = (
        {"": LOCAL_RANK}
    )
    model = AutoModelForCausalLM.from_pretrained(
        finetune_config.base_model,
        load_in_8bit=finetune_config.load_in_8bit,
        load_in_4bit=finetune_config.load_in_4bit,
        torch_dtype=finetune_config.torch_dtype,
        device_map=device_map,
    )

    model.train()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, finetune_config.lora_config)

    if LOCAL_RANK == 0:
        model.print_trainable_parameters()

    train_dataset = finetune_config.train_dataset_config.load()
    eval_dataset = finetune_config.eval_dataset_config.load()

    trainer = Trainer(
        model,
        args=finetune_config.training_arguments,
        data_collator=default_data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    trainer.train()


if __name__ == "__main__":
    run()




