from typing import Tuple

import torch
from datasets import Dataset
from fastchat.serve.cli import SimpleChatIO
from transformers import PreTrainedModel, PreTrainedTokenizerFast, LlamaForCausalLM, LlamaTokenizer
from fastchat.serve.inference import generate_stream
from fastchat.conversation import Conversation, SeparatorStyle

from bert_qa.data import INDEXED_COLUMN
from bert_qa.retriever import Retriever

base_model = "/home/jovyan/workspace/models/vicuna-7b"

def make_model() -> Tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=None,
    )
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
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
        self.system = None
        self.conv = None
        self.output = None

    def load_models(self):
        if self.orig:
            # alread loaded
            return
        model = "pritamdeka/S-PubMedBert-MS-MARCO"
        orig = Dataset.from_json("/home/jovyan/output-heart/pubmed-single/data.jsonl")
        ds = Dataset.from_parquet("/home/jovyan/output-heart/pubmed-single/data-embeddings.parquet")
        ds.load_faiss_index(INDEXED_COLUMN, "/home/jovyan/output-heart/pubmed-single/data-embeddings.faiss")
        retriever = Retriever(context_model=model, question_model=model, load_datasets=False, datasets={'pubmed': ds})
        model, tokenizer = make_model()
        self.orig = orig
        self.ds = ds
        self.retriever = retriever
        self.model = model
        self.tokenizer = tokenizer

    def set_question(self, question):
        self.question = question
        self.load_models()
        results = self.retriever.search(question, None, 3)
        self.results = results[::-1]

    def make_prompt(self, results):
        # system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
        # system += f"The assistant will only use information from the following articles:\n"
        # system = "Given the following extracted parts of a long document and a question, create a final answer. If you don't know the answer, just say that you don't know. Don't try to make up an answer.\n\n"

        system = ""
        for idx, result in enumerate(results):
            print(result.score)
            if result.meta['original_index']:
                text = self.orig[result.meta['original_index']]['text']
            else:
                text = result.text
            result.original_text = text
            # system += f"Article {idx}:\n"
            system += text.strip().replace('\n', '')
            system += "\n\n"
        system += "Please use the above context to answer questions from a curious user.\n\n"
        self.system = system

    def create_conversation(self):
        conv = Conversation(
            name='vicuna_v1.1',
            system=self.system,
            roles=['USER', 'ASSISTANT'],
            messages=[['USER', 'hi'], ['ASSISTANT', None]],
            offset=0,
            sep_style=SeparatorStyle.ADD_COLON_TWO,
            sep=' ',
            sep2='</s>',
            stop_str = None,
            stop_token_ids = None
        )
        self.conv = conv
        self.conv.messages = []

    def start_loop(self, user_input: str):
        conv = self.conv
        conv.append_message(conv.roles[0], user_input)
        conv.append_message(conv.roles[1], None)

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
        print(prompt[-context_len:])
        print(final_out)
        # print("\n", {"prompt": prompt, "outputs": final_out}, "\n")


if __name__ == "__main__":
    chat_loop()
