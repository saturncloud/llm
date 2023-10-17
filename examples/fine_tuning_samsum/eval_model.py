from datasets import load_dataset

from llm.inference.transformer import TransformersEngine

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

from llm.model_configs import ModelConfig
from llm.prompt import Prompt, Message


BASE_MODEL = "meta-llama/Llama-2-7b-hf"
MODEL_MAX_LENGTH=4096
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
)
device_map = "auto"
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    device_map='auto',
    torch_dtype=torch.float16,
    quantization_config=quantization_config
)
model = PeftModel.from_pretrained(model, "/tmp/samsum/final_output")
# model = model.merge_and_unload()

eval_dataset = load_dataset("saturncloud/samsum", split='eval')
prompt_obj = Prompt(
    system_message="Please summarize the following conversation",
    input_template="Conversation: {text}",
    response_template="Summary: {text}"
)
model_config = ModelConfig.from_registry(BASE_MODEL)
engine = TransformersEngine(model, tokenizer, max_length=model_config.max_length)

idx = 0
message = Message(input=eval_dataset[idx]['input'], response="")
prompt = prompt_obj.render([message])
print('**** Here ia thw prompt: ****')
print(prompt)

model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
result = model.generate(**model_input, max_new_tokens=200, do_sample=False)
result = tokenizer.decode(result[0], skip_special_tokens=True)

print('**** Here is the output from the model: ****')
print(result.split('Summary:', 1)[-1])


