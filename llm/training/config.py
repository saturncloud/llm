import os
from dataclasses import dataclass, asdict, field
from typing import Union, Dict, Optional, List, Any
import tempfile

import fsspec.generic
from datasets import load_dataset, load_from_disk, Dataset
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from peft import LoraConfig
import torch
from saturnfs import SaturnFS
import fsspec

from llm.model_configs import ModelConfig
from llm.prompt import Prompt
from llm.qa.prompts import ZeroShotQA, FewShotQA, StandaloneQuestion

fsspec.register_implementation("sfs", SaturnFS)


class ConfigError(Exception):
    pass


experiment_tracking_methods = {}


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

    method: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    report_to: List[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, method, **kwargs):
        return cls(method=method, kwargs=kwargs)

    @classmethod
    def register(cls, name, method) -> None:
        experiment_tracking_methods[name] = method

    def begin(self):
        return experiment_tracking_methods[self.method](self, self.kwargs)


def init_comet_ml(
    config: ExperimentTrackingConfig,
    comet_api_key: Optional[str] = None,
    comet_workspace: Optional[str] = None,
    comet_project_name: Optional[str] = None,
    auto_output_logging=None,
):
    from comet_ml import Experiment

    experiment = Experiment(
        api_key=comet_api_key or os.getenv("COMET_API_KEY"),
        workspace=comet_workspace or os.getenv("COMET_WORKSPACE"),
        project_name=comet_project_name or os.getenv("COMET_PROJECT_NAME"),
        auto_output_logging=auto_output_logging or "native",
    )
    config.report_to = ["comet_ml"]
    return experiment


def init_wandb(
    config: ExperimentTrackingConfig,
    wandb_project: Optional[str] = None,
    wandb_log_model: str = "all",
):
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_log_model:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    config.report_to = ["wandb"]


ExperimentTrackingConfig.register("comet_ml", init_comet_ml)
ExperimentTrackingConfig.register("wandb", init_wandb)


dataset_method_registry = {}


@dataclass
class DatasetConfig:
    # name or path to dataset
    method: str
    kwargs: Dict[str, Any]

    def load(self) -> Dataset:
        method = dataset_method_registry[self.method]
        return method(**self.kwargs)

    @classmethod
    def register(cls, name: str, method: Any):
        dataset_method_registry[name] = method


DatasetConfig.register("load_from_disk", load_from_disk)
DatasetConfig.register("load_dataset", load_dataset)


def default_training_arguments(
    eval_dataset: Optional[DatasetConfig],
    experiment_tracking_config: Optional[ExperimentTrackingConfig] = None,
) -> Dict[str, Any]:
    defaults: Dict[str, Any] = dict(
        bf16=False,
        optim="adamw_torch",
        num_train_epochs=1,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        auto_find_batch_size=False,
        gradient_checkpointing=True,
        logging_strategy="steps",
        logging_steps=10,
        save_steps=30,
        save_strategy="steps",
    )
    if eval_dataset:
        defaults.update(dict(eval_steps=5, evaluation_strategy="steps"))
    if experiment_tracking_config:
        defaults.update({"report_to": [experiment_tracking_config.report_to]})
    return defaults


@dataclass
class FineTuneConfig:
    base_model: str
    train_dataset_config: DatasetConfig
    training_arguments: TrainingArguments
    lora_config: LoraConfig
    experiment_tracking_config: Optional[ExperimentTrackingConfig] = None
    eval_dataset_config: Optional[DatasetConfig] = None

    # we need a local directory on the machine to save files.
    local_output: Optional[str] = None
    # we can additionally copy data to other networked locations.
    additional_output_paths: List[str] = field(default_factory=list)

    torch_dtype: torch.dtype = torch.float16
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    use_gradient_checkpointing: bool = True
    resume_from_checkpoint: bool = True

    @classmethod
    def from_config(cls, **config) -> "FineTuneConfig":
        model_config = ModelConfig.from_registry(config["base_model"])
        default_lora_dict = model_config.default_lora_config.copy()

        if config.get("local_output") is None:
            f = tempfile.mkdtemp()
            config["local_output"] = f

        experiment_tracking_config = load_config(
            ExperimentTrackingConfig, config.pop("experiment_tracking_config", None)
        )
        train_dataset_config = load_config(DatasetConfig, config.pop("train_dataset_config"))
        eval_dataset_config = load_config(DatasetConfig, config.pop("eval_dataset_config"))

        if "torch_dtype" in config:
            config["torch_dtype"] = getattr(torch, config["torch_dtype"])

        training_arguments_dict = default_training_arguments(
            eval_dataset_config, experiment_tracking_config
        )
        training_arguments_dict.update(config.pop("training_arguments", {}))
        training_arguments_dict.setdefault("type", "TrainingArguments")
        training_arguments_dict["output_dir"] = config["local_output"]
        training_arguments = load_config(TrainingArguments, training_arguments_dict)

        lora_config_dict = default_lora_dict
        lora_config_dict.update(config.pop("lora_config", {}))
        lora_config_dict.setdefault("type", "LoraConfig")
        lora_config = load_config(LoraConfig, lora_config_dict)

        if train_dataset_config is None:
            raise ConfigError("train dataset is required")
        return cls(
            training_arguments=training_arguments,
            lora_config=lora_config,
            train_dataset_config=train_dataset_config,
            eval_dataset_config=eval_dataset_config,
            experiment_tracking_config=experiment_tracking_config,
            **config
        )

    def copy_callback(self):
        return CopyToSourcesCallback(self.local_output, self.additional_output_paths)


class CopyToSourcesCallback(TrainerCallback):
    def __init__(self, local_output: str, additional_output_paths: List[str]):
        self.local_output = local_output
        self.additional_output_paths = additional_output_paths

    def rsync(self):
        for path in self.additional_output_paths:
            fsspec.generic.rsync(self.local_output, path)

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """
        Event called at the end of training.
        """
        self.rsync()

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """
        Event called at the end of training.
        """
        self.rsync()


def load_config(cls: type, kwargs: Optional[Dict[str, Any]]) -> Any:
    if kwargs is None:
        return None
    if hasattr(cls, "from_config"):
        return cls.from_config(**kwargs)
    else:
        return cls(**kwargs)


prompt_methods = {}


@dataclass
class PromptConfig:
    method: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def register(cls, name, method) -> None:
        prompt_methods[name] = method

    def load(self, format):
        method = prompt_methods[self.method]
        return method(format=format, **self.kwargs)


PromptConfig.register(Prompt.__name__, Prompt)
PromptConfig.register(ZeroShotQA.__name__, ZeroShotQA)
PromptConfig.register(FewShotQA.__name__, FewShotQA)
PromptConfig.register(StandaloneQuestion.__name__, StandaloneQuestion)


dataset_writers = {}


@dataclass
class DatasetWriterConfig:
    method: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def register(cls, name, method):
        dataset_writers[name] = method

    def save(self, dataset: Dataset):
        method = dataset_writers[self.method]
        method(dataset, **self.kwargs)


DatasetWriterConfig.register(
    "save_to_disk", lambda dataset, dataset_path: dataset.save_to_disk(dataset_path)
)


@dataclass
class DataPrepConfig:
    source_config: DatasetConfig
    base_model: str
    prompt_config: PromptConfig
    dataset_writer_config: DatasetWriterConfig
    # pack examples into chunks of chunksize.
    # examples that are bigger than this will be excluded.
    chunksize: int = 2048
    ignore_index: int = -100
    num_proc: int = 1

    @classmethod
    def from_config(cls, **config: Dict[str, Any]) -> "DataPrepConfig":
        prompt_config = load_config(PromptConfig, config.pop("prompt_config", None))
        source_config = load_config(DatasetConfig, config.pop("source_config"))
        dataset_writer_config = load_config(
            DatasetWriterConfig, config.pop("dataset_writer_config", None)
        )
        return cls(
            prompt_config=prompt_config,
            source_config=source_config,
            dataset_writer_config=dataset_writer_config,
            **config
        )
