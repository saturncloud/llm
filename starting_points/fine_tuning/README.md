# Fine Tuning Samsum

Please read the README for this repository first.

## Dataset preparation

The following section covers the input data format expected by the Saturn Cloud LLM Framework.
It also covers the data processing steps (implemented by this framework) necessary to turn this
dataset into a format suitable for trainig (input_ids, attention_mask and labels)

### Data Format

The Saturn Cloud LLM framework expects data as a HuggingFace dataset, with the following fields
- input: str - The input message
- context: List[str] - A list of additional contexts the LLM will use in a response. Can be omitted
if there is no context.
- response: The response from the LLM

Please prepare your dataset as a HuggingFace dataset. You do not need to upload your data to
HuggingFace. You can store in on local disk, or in S3 if you would like.

### Prompts and Prompt Formats

The Prompt object converts data into text that the model understands. It leverages a PromptFormat
object which has formatting information specific to the model being trained.

### input_ids, labels, and attention_mask

We then pass the prompt through the tokenizer associated with the model you are fine tuning in order
to generate 3 fields.

- input_ids: List[int] - the result of passing text through the tokenizer
- attention_mask: List[int] - ones and zeros
- labels: List[int] - labels are what you are training the model to produce. We generate labels
  by copying input_ids, and masking the values corresponding to the input to be -100 (which the model
  knows to ignore in computing loss). Doing so ensures that we focus on training the model to
  generate the specified output, and not worry about learning how to generate the input.

### Example packing

In training, data must all be padded to the same length in order to be packed into a batch. This
is typically done by padding values. Example packing is an optimization where examples are concatenated
to produce vectors of some maximum length.

For example, assume I want to pack to a maximum length of 9.

I have the following inputs

```python
[
  [1, 100, 100, 2],
  [1, 200, 200, 2]
  [1, 300, 300, 300, 300, 2]
]
```

Without example padding, in order to process these rows in the same batch, they would all need to
be padded

```python
[
  [1, 100, 100, 2, 0, 0],
  [1, 200, 200, 2, 0, 0]
  [1, 300, 300, 300, 300, 2]
]
```

With example packing, we can combine the first 2 in one row.

```python
[
  [1, 100, 100, 2, 1, 200, 200, 2, 0],
  [1, 300, 300, 300, 300, 2, 0, 0, 0]
]
```

### Configuration in this repository


> **Note**
> The Saturn Cloud LLM Framework makes heavy use of configuration so that users can work
with LLMs without having to write lots of code. Sometimes the configuration delegates to other
code/classes, for example
> - load_datest, load_from_disk for referencing HuggingFace datasets
> - UserAssistantFormat, VicunaFormat for PromptFormats
> - ZeroShotQA and FewShotQA classes for Prompts
>
> These configurations are specified with a `method` and a `kwargs` value. `method` is a string that
has been registered against an existing python function in code. You can also call methods that
we haven't registerd with the following syntax: `path.to.module::name`. The kwargs entry is a dictionary of
> parameters that the method expects.

### Running dataset preparation.

```
$ python llm/training/dataprep.py starting_points/fine_tuning/dataprep_train.yaml
```

In addition to formatting your dataset, you will need to configure some information 
in `dataprep_train.yaml` (which has been stubbed out a bit in this repository)

- source_config: This is the configuration for the hugging face datset you've configured. This 
configuration expects a method (`load_from_disk`, `load_dataset` or any other pointer to a method
with the syntax `path.to.module::name` ). It also expects a field called `kwargs` which is a 
dictionary of parameters to that method.
- base_model: The ID of the model you are going to fine tune. such as `meta-llama/Llama-2-7b-hf`
- prompt_config: Configuration for the specific prompt object that will be used. 
- dataset_writer_config: Configuration for writing the dataset 

The default prompt_config is probably sufficient for what you were doing but we recommend 
reading the section on [Prompts](../../README.md#prompts) and 
[PromptFormats](../../README.md#prompt-format) and creating a Prompt explicilty.

## Fine Tuning

```
$ python llm/training/finetune.py starting_points/fine_tuning_samsum/finetune.yaml
```

The `finetune.py` script will fine tune the base model using the data generated in the previous step.
The specifics of the fine tuning job are defined in `finetune.yaml`

- base_model specifies which model and tokenizer will be used for fine tuning.
- train_dataset_config specifies the training dataset
- eval_dataset_config specifies the evaluation dataset (optional) 
- training_arguments are passed into the HuggingFace `TrainingArguments` object.
  We also define our own default parameters that are suitable for most users.
  This can be found in `llm.training.config::default_training_arguments`
- local_output - this directory is set as the `output_dir` in `TrainingArguments`.
  This means that all checkpoints will be saved to this location. In addition, once
  the training job is complete, the model will be saved to `${local_output}/final_output`
- additional_output_paths - a list of additional output paths. The contents of
  local_output will be copied (using `fsspec.generic.rsync`) to this location every time
  a checkpoint is saved, and when the final training run is complete. You can use
  any protocol that fsspec understands, including things like `s3://` to save to S3.
- load_in_4bit and load_in_8bit sets up 4bit and 8bit quantization. If you do not
  which to use these flags, you can set `quantization_config` which should be arguments
  to a `BitsAndBytesConfig` object. for 4bit quantization, we override some of the
  default `BitsAndBytesConfig` options.
 