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
Prompt(
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


## Fine Tuning

The following command will run the fine tuning script.
```
$ python llm/training/finetune.py examples/fine_tuning_samsum/finetune.yaml
```

Please see the 
[documentation to dig into the fine tuning workflow](../../llm/training/README.md#fine-tuning)

The specifics of the fine tuning job are defined in `finetune.yaml`

```yaml
base_model: "meta-llama/Llama-2-7b-hf"
train_dataset_config:
  method: load_from_disk
  kwargs:
    dataset_path: "/tmp/samsum-train"
training_arguments:
  logging_steps: 2
  max_steps: 100
local_output: /tmp/samsum
load_in_4bit: true
# load_in_8bit: true
```
The parameters are discussed in the documentation, however a few things to call out.

- You can configure any of the HuggingFace training arguments,
we set max_steps to 100, and log every 2 steps so that the example runs and produces output quickly.
- This example uses 4 bit quantization.

This should run in about 45 minutes on an A10g. Thanks to the 4 bit quantization, it should also
run on machines with 16GB of GPU memory.

## Evaluating the model
This framework has other infrastructure that can facilitate batch inference, but for simplicity
we do not use that here.

### The Base Model
The following script will loads the base model (llama2) and pass the input from our dataset 
into it. 

```bash
$ python examples/fine_tuning_samsum/baseline.py
```

As expected, it performs poorly. For the given prompt:

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

We get the following nonsensical output (yours may differ depending on randomness). Instead of
summarizing the conversation, the model just continues to generate more of it.

```text
A: Hi Tom, are you busy tomorrow’s afternoon?
B: I’m pretty sure I am. What’s up?
A: Can you go with me to the animal shelter?
B: What do you want to do?
A: I want to get a puppy for my son.
B: That will make him so happy.
A: Yeah, we’ve discussed it many times. I think he’s ready now.
B: That’s good. Raising a dog is a tough issue. Like having a baby ;-) 
A: I'll get him one of those little dogs.
B: One that won't grow up too big;-)
A: And eat too much;-))
B: Do you know which one he would like?
A: Oh, yes, I took him there last Monday. He showed me one that
```


### The fine tuned model

The following does the same, except with the fine tuned model
```bash
$ python examples/fine_tuning_samsum/eval_model.py 
```

The only change is the addition of the following line:

```python
model = PeftModel.from_pretrained(model, "/tmp/samsum/final_output")
```

which loads the model we fine tuned on top of the base model.  
The output is much more sensible this time:

```text
Tom and his friend are going to the animal shelter to get a puppy for Tom's son.
```