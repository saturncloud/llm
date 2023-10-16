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
User: summarize the following dialogue:
A: Hi Tom, are you busy tomorrow’s afternoon?
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
Assistant:
```

We get the following nonsensical output (yours may differ depending on randomness)

```text
 What is the meaning of the word “fungible” in the following sentence?
“The fungible commodity was sold at a price lower than its average cost.”
Assistant: What does the word “fungible” mean in the following sentence?
“The fungible commodity was sold at a price lower than its average cost.”
Assistant: What is the meaning of the word “fungible” in the following sentence?
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
The dialogue is about a boy who wants to get a dog. The boy's mother is going to the animal shelter with him.
```