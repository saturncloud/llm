# Fine Tuning Samsum

Please read the README for this repository first.

## The Dataset
The SAMSum dataset contains about 16k messenger-like conversations with summaries. 
This examples loads data from huggingface `saturncloud/samsum` which is based off of the original
`samsum` dataset, but processed into messages as follows:

input:

```
Summarize the following dialogue:	
Amanda: I baked cookies. Do you want some? 
Jerry: Sure! 
Amanda: I'll bring you tomorrow :-)
```

response:
```
Amanda baked cookies and will bring Jerry some tomorrow.
```
**We expect an input dataset to this workflow to contain input, response, and contexts.** 
Contexts are optional. When specified, they should be a list of strings

## Datset preparation

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

### Running dataset preparation.

```
$ python llm/training/dataprep.py examples/fine_tuning_samsum/dataprep_train.yaml
```

The `dataprep.py` script will generate prompts, tokenize them, and pack/pad examples and write
the resulting dataset to `/tmp/samsum_train`

The specifics of the job are defined in `examples/fine_tuning_samsum/dataprep_train.yaml`

```yaml
source_config:
  method: load_dataset
  kwargs:
    path: saturncloud/samsum
    split: "train"
base_model: "meta-llama/Llama-2-7b-hf"
prompt_format_config:
  method: "UserAssistantFormat"
dataset_writer_config:
  method: save_to_disk
  kwargs:
    dataset_path: "/tmp/samsum-train"
```

- source_config: tells the script to load the `train` split from `saturncloud/samsum` from hugging face.
- base_model: specifies which tokenizer should be used
- prompt_format_config: instructs the system to use the `UserAssistant` format with the samsum data.
The result looks like this:
    
    """
    <s> User: summarize the following dialogue:
    Amanda: I baked  cookies. Do you want some?
    Jerry: Sure!
    Amanda: I'll bring you tomorrow :-)
    Assistant: Amanda baked cookies and will bring Jerry some tomorrow.</s>
    """
- dataset_writer_config writes data to /tmp/samsum-train when complete

## Fine Tuning

