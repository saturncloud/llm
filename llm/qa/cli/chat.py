import os
import runpy
import sys
from typing import Optional
import click

from llm import settings
from llm.model_configs import ModelConfig, VicunaConfig
from llm.qa.embedding import DEFAULT_EMBEDDING_MODEL, QAEmbeddings
from llm.qa.session import QASession
from llm.qa.vector_store import DatasetVectorStore
from llm.utils.data import load_data


@click.group(name="chat", short_help="Commands for chatting over data")
def chat_cli():
    pass


@chat_cli.command("cmdline", short_help="Conversational question answering from semantic search")
@click.argument("dataset-path", required=True, envvar="QA_DATASET_PATH")
@click.option(
    "--model-id",
    help="Chat model ID for prompt formatting.",
    default=VicunaConfig.model_id,
    envvar="DEFAULT_MODEL_ID",
)
@click.option(
    "--context-model",
    help="Model name or path for context embedding",
    default=DEFAULT_EMBEDDING_MODEL,
    envvar="QA_CONTEXT_MODEL",
)
@click.option(
    "--dataset-type",
    help="Input file type. Defaults to file extension.",
    default=None,
    envvar="QA_INPUT_TYPE",
)
@click.option(
    "--index-path",
    help="Path to a pre-built FAISS index over the dataset",
    default=None,
    envvar="QA_INDEX_PATH",
)
@click.option(
    "--rephrase", is_flag=True, help="Rephrase the question with context from previous messages"
)
@click.option("--max-new-tokens", default=256, type=int, help="Max new generated tokens")
@click.option("--temperature", default=0.7, type=float, help="Logit sampling temperature")
@click.option("--top-p", default=1.0, type=float, help="Logit sampling Top P")
def cmdline_cli(
    dataset_path: str,
    dataset_type: Optional[str],
    index_path: Optional[str],
    model_id: str,
    context_model: str,
    rephrase: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    dataset = load_data(dataset_path, dataset_type)
    if index_path is None:
        _index_path = dataset_path.rsplit(".", 1)[-1] + ".faiss"
        if os.path.isfile(_index_path):
            index_path = _index_path

    model_config = ModelConfig.from_registry(model_id)
    vector_store = DatasetVectorStore(dataset, QAEmbeddings(context_model), index_path=index_path)
    qa_session = QASession.from_model_config(model_config, vector_store)

    while True:
        input_text = input("Question: ")
        qa_session.append_question(input_text)
        search_query = input_text
        if rephrase:
            search_query = qa_session.rephrase_question(
                input_text, temperature=temperature, top_p=top_p
            )
        qa_session.search_context(search_query)

        prev_output = ""
        print("Answer: ", end="", flush=True)
        for output_text in qa_session.stream_answer(
            input_text, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p
        ):
            new_output = output_text[len(prev_output) :]
            prev_output = output_text
            print(new_output, end="", flush=True)
        print()


if __name__ == "__main__":
    chat_cli()
