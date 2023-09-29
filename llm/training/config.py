import os
from dataclasses import dataclass, asdict, field
from typing import Union, Dict, Optional, List, Any
import tempfile

import fsspec.generic
from datasets import load_dataset, load_from_disk
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from peft import LoraConfig

from llm.model_configs import ModelConfig


configs = {}


class ConfigError(Exception):
    pass


def register(config_class):
    configs[config_class.__name__] = config_class


@dataclass
class ExperimentTrackingConfig:
    """
    This configuration object defines values pertaining to different
    experiment tracking solutions. In many cases, this config
    will be responsible for setting environment variables.

    This isn't always necessary,most users can set configuration variables outside
    of this config, but this exists to make it easy to configure experiment tracking in config.
    As a result - most of the values will be optional
    """
    def set_env(self):
        obj = asdict(self)
        for k, v in obj:
            if v is not None:
                os.environ[k] = v
    @property
    def report_to(self):
        raise NotImplementedError


@dataclass
class CometMLConfig(ExperimentTrackingConfig):
    COMET_API_KEY: Optional[str] = None
    COMET_WORKSPACE: Optional[str] = None
    COMET_PROJECT_NAME: Optional[str] = None

    @property
    def report_to(self):
        return "comet_ml"


@dataclass
class WandBConfig(ExperimentTrackingConfig):
    WANDB_API_KEY: Optional[str] = None
    WANDB_BASE_URL: Optional[str] = None
    WANDB_PROJECT: Optional[str] = None

    @property
    def report_to(self):
        return "wandb"

@dataclass
class LoadDatasetConfig:
    # name or path to dataset
    path: str
    def load(self):
        return load_dataset(self.path)


@dataclass
class LoadFromDiskConfig:
    # path to dataset. can be fsspec path
    dataset_path: str
    def load(self):
        return load_from_disk(self.dataset_path)


def default_training_arguments(eval_dataset: Optional[Union[LoadDatasetConfig, LoadFromDiskConfig]], experiment_tracking_config: Optional[ExperimentTrackingConfig] = None) -> Dict[str, Any]:
    defaults = dict(
        bf16=False,
        optim="adamw_torch",
        num_train_epochs=1,
        gradient_accumulation_steps=2,
        auto_find_batch_size=True,
        gradient_checkpointing=True,
        logging_strategy="steps",
        logging_steps=10,
        max_steps=240,
        save_steps=30,
        save_strategy="steps",
    )
    if eval_dataset:
        defaults.update(dict(
            eval_steps=5,
            evaluation_strategy="steps"
        ))
    if experiment_tracking_config:
        defaults.update(dict(
            report_to=[experiment_tracking_config.report_to]
        ))
    return defaults


@dataclass
class FineTuneConfig:
    base_model: str
    train_dataset_config: Union[LoadDatasetConfig, LoadFromDiskConfig]
    training_arguments: TrainingArguments
    lora_config: LoraConfig
    experiment_tracking_config: Optional[Union[CometMLConfig, WandBConfig]] = None
    eval_dataset_config: Optional[Union[LoadDatasetConfig, LoadFromDiskConfig]] = None

    # we need a local directory on the machine to save files.
    local_output: Optional[str] = None
    # we can additionally copy data to other networked locations.
    additional_output_paths: List[str] = field(default_factory=list)

    torch_dtype: str = "float16"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    use_gradient_checkpointing: bool = True

    @classmethod
    def from_config(cls, **config) -> "FineTuneConfig":
        model_config = ModelConfig.from_registry(config['base_model'])
        default_lora_dict = model_config.default_lora_config.copy()

        if config.get('local_output') is None:
            f = tempfile.mkdtemp()
            config['local_output'] = f

        experiment_tracking_config = load_config(config.pop('experiment_tracking_config'))
        train_dataset_config = load_config(config.pop('train_dataset_config'))
        eval_dataset_config = load_config(config.pop('eval_dataset_config'))

        training_arguments_dict = default_training_arguments(eval_dataset_config)
        training_arguments_dict.update(config.pop('training_arguments', {}))
        training_arguments_dict.setdefault("type", "TrainingArguments")
        training_arguments_dict['output_dir'] = config["local_output"]
        training_arguments = load_config(training_arguments_dict)

        lora_config_dict = default_lora_dict
        lora_config_dict.update(config.pop('lora_config', {}))
        lora_config_dict.setdefault("type", "LoraConfig")
        lora_config = load_config(lora_config_dict)
        if train_dataset_config is None:
            raise ConfigError("train dataset is required")
        return cls(training_arguments=training_arguments, lora_config=lora_config, train_dataset_config=train_dataset_config, eval_dataset_config=eval_dataset_config, output_path=output_path, **config)

    def copy_callback(self):
        return CopyToSourcesCallback(self.local_output, self.additional_output_paths)


class CopyToSourcesCallback(TrainerCallback):
    def __init__(self, local_output: str, additional_output_paths: List[str]):
        self.local_output = local_output
        self.additional_output_paths = additional_output_paths

    def rsync(self):
        for path in self.additional_output_paths:
            fsspec.generic.rsync(self.local_output, path)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """
        self.rsync()

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """
        self.rsync()


register(CometMLConfig)
register(WandBConfig)
register(LoadDatasetConfig)
register(LoadFromDiskConfig)
register(FineTuneConfig)
register(TrainingArguments)
register(LoraConfig)


def load_config(config: Dict[str, Any]) -> Optional[Union[CometMLConfig, WandBConfig, LoadFromDiskConfig, LoadDatasetConfig, FineTuneConfig, LoraConfig]]:
    if config is None:
        return None
    config_class = configs[config.pop('type')]
    return config_class(**config)
