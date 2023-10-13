import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "/home/jovyan/tmp/llama2-dummy-labels-concat-4bit/final_output"
BASE_MODEL = "meta-llama/Llama-2-7b-hf"


tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map='cpu',
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(model, model_id)
model = model.merge_and_unload()
model.save_pretrained("/home/jovyan/tmp/llama2-dummy-labels-concat-4bit/merged")
tokenizer.save_pretrained("/home/jovyan/tmp/llama2-dummy-labels-concat-4bit/merged")
