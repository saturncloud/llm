# Fine Tuning Samsum

Please read the README for this repository first.

## Datset preparation

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

In order to convert the data into a data format suitable for fine tuning, we must first convert
our data into an actual prompt. To do so, we use a `Prompt` object (specified in your configuration)
to render each row (input, response, contexts) of the hugging face dataset into a string.

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

The Prompt Format is the format that was used to train the model. 
It is a good idea to use the Prompt Format for a given model, but sometimes not essential.

For example the Llama 2 chat model expects prompts to follow this style:

"""
<s>[INST] <<SYS>>
{system_message}
<</SYS>>

{input} [/INST] {response} </s>
"""

Whereas Vicuna expects prompts to follow this style:

"""
<s> {system_message}
USER: {input}
ASSISTANT: {response}
</s>
"""

The Prompts include specific system messages, examples (for few-shot learning)
and some other formatting. Prompts can be mixed with different PromptFormats. 

