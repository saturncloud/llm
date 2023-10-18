# Fine Tuning / Training scripts

This section contains the scripts used in Fine Tuning LLMs. 
The General workflow for fine tuning LLMS with the Saturn Cloud LLM Framework is:

1. Creating a Hugging Face Dataset in the correct format. This can be on disk - it does not need
   to be uploaded to the Hub
2. You run the `dataprep.py` script which turns your input data into text based prompts, as well as
   data actually used in training, input_ids, labels and the attention_mask
3. You run the finetune.py script to fine tune the model.


## Data Preparation Steps

The following section covers the input data format expected by the Saturn Cloud LLM Framework.
It also covers the data processing steps (implemented by this framework) necessary to turn this
dataset into a format suitable for trainig (input_ids, attention_mask and labels)

### Data Format

The Saturn Cloud LLM framework expects data as a HuggingFace dataset, with the following fields
- input: str - The input message
- contexts: List[str] - A list of additional contexts the LLM will use in a response. Can be omitted
  if there is no context.
- response: The response from the LLM

Please prepare your dataset as a HuggingFace dataset. You do not need to upload your data to
HuggingFace. You can store in on local disk, or in S3 if you would like.

### Prompts and Prompt Formats

The Prompt object converts data into text that the model understands. It leverages a PromptFormat
object which has formatting information specific to the model being trained. Please read 
through the documentation on [Prompts](../../README.md#prompts) and
[PromptFormats](../../README.md#prompt-format).


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

## Running dataset preparation.

```
$ python llm/training/dataprep.py path-to-config.yaml
```

In addition to formatting your dataset, you will need to write a configuration file. Some relevant
inputs:

- source_config: This is the configuration for the hugging face dataset you've configured. Some 
  examples:
  - ```yaml
    source_config:
      method: load_dataset
      kwargs:
        path: saturncloud/samsum
        split: "eval"
    ```
  - ```yaml
      method: load_from_disk
      kwargs:
        dataset_path: "/tmp/train"
    ```
  - ```yaml
      method: load_from_disk
      kwargs:
        dataset_path: "s3://my-data/train"
    ```    
- base_model: The ID of the model you are going to fine tune. such as `meta-llama/Llama-2-7b-hf`
- prompt_config: Configuration for the specific prompt object that will be used.
- dataset_writer_config: Configuration for writing the dataset

The default prompt_config is probably sufficient for what you were doing but we recommend
reading the section on [Prompts](../../README.md#prompts) and
[PromptFormats](../../README.md#prompt-format) and creating a Prompt explicilty.


## Fine Tuning

```
$ python llm/training/finetune.py config.yaml
```

The `finetune.py` script will fine tune the base model using the data generated in the previous step.
The specifics of the fine tuning job are defined in `config.yaml`

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
