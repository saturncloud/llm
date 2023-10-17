# Fine Tuning Samsum

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

## General Workflow

The General workflow for fine tuning LLMS with the Saturn Cloud LLM Framework is:
1. Creating a Hugging Face Dataset in the correct format. This can be on disk - it does not need 
  to be uploaded to the Hub
2. You run the `dataprep.py` script which turns your input data into text based prompts, as well as
   data actually used in training, input_ids, labels and the attention_mask
3. You run the finetune.py script to fine tune the model.

## Datset preparation

The following command will run the data preparation steps.

```bash
$ python llm/training/dataprep.py examples/fine_tuning_samsum/dataprep_train.yaml
```
which will use the following configuration:

```yaml
source_config:
  method: load_dataset
  kwargs:
    path: saturncloud/samsum
    split: "train"
base_model: "meta-llama/Llama-2-7b-hf"
prompt_config:
  method: Prompt
  kwargs:
    system_message: "Please summarize the following conversation"
    input_template: "Conversation: {text}"
    response_template: "Summary: {text}"
dataset_writer_config:
  method: save_to_disk
  kwargs:
    dataset_path: "/tmp/samsum-train"
```

- `source_config` tells us where to load the data
- `base_model` this is the model we are fine tuning. In this phase, the model isn't relevant but
we use this to load the Tokenizer.
- `prompt_config` more on this later, but this Prompt turns our data into text we want the LLM
to generate.
- `dataset_writer_config` where we save the output.

To understand what's going on, Please read through the 
[section on data preparation](../../llm/training/README.md#data-preparation-steps)
as well as the documentation on [Prompts](../../README.md#prompts) and
[PromptFormats](../../README.md#prompt-format). They are short.

This example does not use any context information, only inputs and responses. We are training 
`meta-llama/Llama-2-7b-hf` 
which does not have a PromptFormat. The Prompt we are using is as follows:

```python
DefaultPrompt(
    system_message="Please summarize the following conversation",
    input_template="Conversation: {text}",
    response_template="Summary: {text}"
)
```

This results in rendered full text output as follows:

```text
Please summarize the following conversation
Conversation: A: Hi Tom, are you busy tomorrow’s afternoon?
B: I’m pretty sure I am. What’s up?
A: Can you go with me to the animal shelter?.
B: What do you want to do?
A: I want to get a puppy for my son.

Summary: Tom and his friend are going to the animal shelter to get a puppy for Tom's son.
```

The data prep step will load data from HuggingFace (`saturncloud/samsum`), generate full 
text prompts based on the above, convert that full text into tokenized outputs (`input_ids`, 
`labels` and `attention_mask`) and pack all the examples to buffers of length 2048. The results
are saved to `/tmp/samsum-train`



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

```
$ python llm/training/finetune.py examples/fine_tuning_samsum/finetune.yaml
```

The `finetune.py` script will fine tune the base model using the data generated in the previous step.
The specifics of the fine tuning job are defined in `finetune.yaml`

```yaml
base_model: "meta-llama/Llama-2-7b-hf"
train_dataset_config:
  method: load_from_disk
  kwargs:
    dataset_path: "/tmp/samsum-train"
training_arguments:
  per_device_train_batch_size: 3
  per_device_eval_batch_size: 3
  logging_steps: 2
  max_steps: 30
local_output: /tmp/samsum
load_in_4bit: true
```

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

## Evaluating the model

### The Base Model
This framework has other infrastructure that can facilitate batch inference, but for simplicity
we do not use that here. 

```bash
$ python examples/fine_tuning_samsum/baseline.py
```

This script loads the base model (llama2) and passes the input into it. As expected, it performs 
poorly. For the given prompt:

```text
Please summarize the following conversation
Conversation: A: Hi Tom, are you busy tomorrow’s afternoon?
B: I’m pretty sure I am. What’s up?
A: Can you go with me to the animal shelter?.
B: What do you want to do?
A: I want to get a puppy for my son.
B: That will make him so happy.
A: Yeah, we’ve discussed it many times. I think he’s ready now.
B: That’s good. Raising a dog is a tough issue. Like having a baby ;-) 
A: I'll get him one of those little dogs.
B: One that won't grow up too big;-)
A: And eat too much;-))
B: Do you know which one he would like?
A: Oh, yes, I took him there last Monday. He showed me one that he really liked.
B: I bet you had to drag him away.
A: He wanted to take it home right away ;-).
B: I wonder what he'll name it.
A: He said he’d name it after his dead hamster – Lemmy  - he's  a great Motorhead fan :-)))
Summary:
```

We get the following nonsensical output (yours may differ depending on randomness)

```text
  Tom and Alice went to the animal shelter to get a puppy for Alice's son. Alice's son wanted a small dog. Tom and Alice talked about the difficulties of raising a dog. Alice's son took a dog to the shelter last Monday. He liked the dog very much. Alice's son wanted to take the dog home right away. Alice's son named the dog after his dead hamster, Lemmy. Alice's son is a great Motorhead fan.
A: Hi Tom, are you busy tomorrow's afternoon?
B: I'm pretty sure I am. What's up?
A: Can you go with me to the animal shelter?
B: What do you want to do?
A: I want to get a puppy for my son.
B: That will make him so happy.
A: Yeah, we've discussed it many times. I think he's ready now.
```


### The fine tuned model

```bash
$ python examples/fine_tuning_samsum/eval_model.py 
```

This script will evaluate the prompt on the fine-tuned model. The only change is the addition of
the following line:

```python
model = PeftModel.from_pretrained(model, "/tmp/samsum/final_output")
```

The output is much more sensible this time:

```text
Tom and his friend are going to the animal shelter to get a puppy for Tom's son.
```