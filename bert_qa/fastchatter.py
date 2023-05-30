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


if __name__ == "__main__":
    chat_loop()
