# Fine Tuning

This starting point is designed to help you run the LLM fine-tuning workflow on your own dataset.  We have stubbed
out all the boilerplate, and you are responsible for connecting this code to your data. If you are new to fine-tuning
LLMs it might be beneficial to run through our 
[example which fine tunes Llama2 to learn how to summarize conversations.](../../examples/fine_tuning_samsum/README.md)

> [!NOTE]
> The commands in this file use the `$LLM_ROOT` environment variable which should
> point to the location where you have checked out this repository. You must also
> ensure that `$LLM_ROOT` is on your `$PYTHONPATH`

## General Workflow

The General workflow for fine tuning LLMS with the Saturn Cloud LLM Framework is:
1. Creating a Hugging Face Dataset in the correct format. This can be on disk - it does not need
   to be uploaded to the Hub
2. You run the `dataprep.py` script which turns your input data into text based prompts, as well as
   data actually used in training, input_ids, labels and the attention_mask
3. You run the finetune.py script to fine tune the model.

## Creating your Dataset

The Saturn Cloud LLM framework expects data as a HuggingFace dataset, with the following fields

- input: `str` - The input message
- context: `List[str]` - A list of additional contexts the LLM will use in a response. Can be omitted if there is no context.
- response: `str` - The response from the LLM

Please prepare your dataset as a HuggingFace dataset. You do not need to upload your data to HuggingFace. 
You can store in on local disk, or in S3 if you would like.


## Dataset preparation

To understand the workflow, Please read through the
[section on data preparation](../../llm/training/README.md#data-preparation-steps)
as well as the documentation on [Prompts](../../README.md#prompts) and
[PromptFormats](../../README.md#prompt-format). 

To run the dataprep, please execute

```bash
$ python -m llm.training.dataprep ${LLM_ROOT}/starting_points/fine_tuning/dataprep_train.yaml
```

You will have to flush out the content of `dataprep_train.yaml`. At a minimum you must fill out 
the `source_config` and the `base_model` - which is used to load the appropriate tokenizer. You 
can also customize tne `prompt_config`. With that configuration, the dataset used for fine tuning
be saved to `/tmp/train`. This will be a dataset with the following features

- input_ids
- labels
- attention_mask

We also populate `full_text` which is the text used in the generation of the above, though it will not 
be used in fine tuning. 
Please see our [documentation on data preparation](../../llm/training/README.md#data-preparation-steps) 
to see the rest of the parameters you can adjust. 

## Fine Tuning

To run the fine tuning, please execute

```bash
$ python -m llm.training.finetune ${LLM_ROOT}/starting_points/fine_tuning/finetune.yaml
```

You will have to flush out the content of `finetune.yaml`. At a minimum you must fill out the `base_model` 
to specify which model you are fine tuning (should be the same as the previous section). The configuration
also expects that you saved your dataprep to `/tmp/train`. If you changed the output in the previous step,
you should update this configuration accordingly. This configuration sets `max_steps` to 100. You can
override this, as well as any other HuggingFace training argument in the configuration. 

Please see our [documentation on fine tuning](../../llm/training/README.md#fine-tuning) 
to see the rest of the parameters you can adjust. 