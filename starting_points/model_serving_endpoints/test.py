import os

import requests

from llm.model_configs import ModelConfig
from llm.prompt import Prompt, Message
model_config = ModelConfig.from_registry(os.environ["MODEL_PATH"])
prompt = Prompt.from_model_config(model_config)
input_str = prompt.render([Message(input='Please write me a story about a man named fred')])

print(input_str)

base_url = "http://localhost:8000/api/inference"

response = requests.post(base_url, json={'input': 'Please write me a story about a man named fred'})
print(response)
print(response.json())