from typing import Tuple

import torch
from datasets import Dataset
from fastchat.serve.cli import SimpleChatIO
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, LlamaForCausalLM, LlamaTokenizer
from fastchat.serve.inference import generate_stream
from fastchat.conversation import Conversation, SeparatorStyle

from bert_qa.data import INDEXED_COLUMN
from bert_qa.retriever import Retriever
from bert_qa import model_configs

base_model = "/home/jovyan/workspace/models/vicuna-7b"


def make_model() -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=None,
    )
    # Llama fast tokenizer is not good
    use_fast = not isinstance(model, LlamaForCausalLM)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=use_fast)
    return model, tokenizer


def chat_loop():
    debug = False
    model = "pritamdeka/S-PubMedBert-MS-MARCO"
    orig = Dataset.from_json("/home/jovyan/output-cardiology/pubmed/data.jsonl")
    ds = Dataset.from_parquet("/home/jovyan/output-cardiology/pubmed/data-embeddings2.parquet")
    ds.load_faiss_index(INDEXED_COLUMN, "/home/jovyan/output-cardiology/pubmed/data-embeddings.faiss")
    retriever = Retriever(context_model=model, question_model=model, load_datasets=False, datasets={'pubmed': ds})
    model, tokenizer = make_model()
    chatio = SimpleChatIO()

    question = chatio.prompt_for_input("USER")
    results = retriever.search(question, None, 20)
    system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
    system += f"The system has access to the following articles:\n"

    for idx, result in enumerate(results[::-1]):
        print(result.score)
        if result.meta['original_index']:
            text = orig[result.meta['original_index']]['text']
        else:
            text = result.text
        system += f"Article {idx}:\n"
        system += text
        system += "\n"
    conv = Conversation(
        name='vicuna_v1.1',
        system=system,
        roles = ['USER', 'ASSISTANT'],
        messages = [['USER', 'hi'], ['ASSISTANT', None]],
        offset = 0,
        sep_style = SeparatorStyle.ADD_COLON_TWO,
        sep = ' ',
        sep2 = '</s>',
        stop_str = None,
        stop_token_ids = None
    )
    init = True
    while True:
        if init:
            inp = question
            init = False
        else:
            try:
                inp = chatio.prompt_for_input(conv.roles[0])
            except EOFError:
                inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        gen_params = {
            "model": base_model,
            "prompt": prompt,
            "temperature": 0.7,
            "repetition_penalty": 1.0,
            "max_new_tokens": 512,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }
        chatio.prompt_for_output(conv.roles[1])
        output_stream = generate_stream(model, tokenizer, gen_params, 0)
        outputs = chatio.stream_output(output_stream)
        conv.messages[-1][-1] = outputs.strip()

        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


class StreamlitChatLoop:
    def __init__(self):
        self.orig = None
        self.ds = None
        self.retriever = None
        self.model = None
        self.tokenizer = None
        self.output = None
        self.results = []
        self.model_config = model_configs.VICUNA
        self.conv = self.model_config.create_conversation()
        self.load_models()

    def load_models(self):
        if self.orig:
            # alread loaded
            return
        model = "pritamdeka/S-PubMedBert-MS-MARCO"
        orig = Dataset.from_json("/home/jovyan/output-heart/pubmed-single/data.jsonl")
        ds = Dataset.from_parquet("/home/jovyan/output-heart/pubmed-single/data-embeddings.parquet")
        ds.load_faiss_index(INDEXED_COLUMN, "/home/jovyan/output-heart/pubmed-single/data-embeddings.faiss")
        self.retriever = Retriever(context_model=model, question_model=model, load_datasets=False, datasets={'pubmed': ds})
        self.model, self.tokenizer = self.model_config.load()
        self.orig = orig
        self.ds = ds

    def append_question(self, question: str):
        self.conv.append_message(self.conv.roles[0], question)
        self.conv.append_message(self.conv.roles[1], None)

    def search_results(self, question: str):
        results = self.retriever.search(question, None, 3)
        self.results = results[::-1]
        self.make_prompt(self.results)

    def get_chat(self) -> str:
        messages = [f'{role}: {"" if message is None else message}' for role, message in self.conv.messages]
        return "\n\n".join(messages)

    def make_prompt(self, results):
        context = ""
        for idx, result in enumerate(results):
            print(result.score)
            if result.meta['original_index']:
                text = self.orig[result.meta['original_index']]['text']
            else:
                text = result.text
            result.original_text = text
            text = text.strip().replace("\n", "")
            context += f"{self.model_config.context_label}: {text}\n\n"

        prompt_kwargs = {}
        if "roles" in self.model_config.default_prompt.inputs:
            prompt_kwargs["roles"] = self.conv.roles
        if "context_label" in self.model_config.default_prompt.inputs:
            prompt_kwargs["context_label"] = self.model_config.context_label

        self.conv.system = self.model_config.render_prompt(context=context, **prompt_kwargs)

    def loop(self):
        conv = self.conv
        print(conv.messages)
        prompt = conv.get_prompt()

        gen_params = {
            "model": base_model,
            "prompt": prompt,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.0,
            "max_new_tokens": 512,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }
        context_len = 2048
        output_stream = generate_stream(self.model, self.tokenizer, gen_params, 0, context_len=context_len)

        pre = 0
        output_text = ""
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            output_text = " ".join(output_text)
            yield output_text
        final_out = output_text
        conv.messages[-1][-1] = final_out.strip()
        print('**DEBUG**')
        print(prompt)
        print(final_out)
        # print("\n", {"prompt": prompt, "outputs": final_out}, "\n")


if __name__ == "__main__":
    chat_loop()
