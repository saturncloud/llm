import os
import runpy
import sys
from typing import Optional
import click

from llm import model_configs, settings
from llm.qa.embedding import DEFAULT_MODEL, QAEmbeddings
from llm.qa.session import QASession
from llm.qa.vector_store import DatasetVectorStore
from llm.utils.data import load_data


@click.group(name="chat", short_help="Commands for chatting over data")
def chat_cli():
    pass


@chat_cli.command("cmdline", short_help="Conversational question answering from semantic search")
@click.argument("dataset-path", required=True, envvar="QA_DATASET_PATH")
@click.option("--dataset-type", help="Input file type. Defaults to file extension.", default=None, envvar="QA_INPUT_TYPE")
@click.option("--index-path", help="Path to a pre-built FAISS index over the dataset", default=None, envvar="QA_INDEX_PATH")
@click.option("--context-model", help="Model name or path for context embedding", default=DEFAULT_MODEL, envvar="QA_CONTEXT_MODEL")
@click.option("--rephrase", is_flag=True, help="Rephrase the question with context from previous messages")
def cmdline_cli(dataset_path: str, dataset_type: Optional[str], index_path: Optional[str], context_model: str, rephrase: bool):
    dataset = load_data(dataset_path, dataset_type)
    if index_path is None:
        _index_path = dataset_path.rsplit(".", 1)[-1] + ".faiss"
        if os.path.isfile(_index_path):
            index_path = _index_path

    model_config = model_configs.VICUNA_7B
    vector_store = DatasetVectorStore(dataset, QAEmbeddings(context_model), index_path=index_path)
    qa_session = QASession.from_model_config(model_config, vector_store)

    while True:
        input_text = input("Question: ")
        qa_session.append_question(input_text)
        search_query = input_text
        if rephrase:
            search_query = qa_session.rephrase_question(input_text)
        qa_session.search_context(search_query)

        prev_output = ""
        print("Answer: ", end="", flush=True)
        for output_text in qa_session.stream_answer(input_text):
            new_output = output_text[len(prev_output):]
            prev_output = output_text
            print(new_output, end="", flush=True)
        print()


@chat_cli.group("streamlit", short_help="Run a simple Streamlit QA application", invoke_without_command=True)
@click.pass_context
def streamlit_cli(ctx):
    if ctx.invoked_subcommand is None:
        # Default to local transformers
        ctx.forward(transformers_backend)


@streamlit_cli.command("transformers", short_help="Local transformers engine")
@click.option("--model-id", help="Chat model ID.", default=model_configs.VICUNA_7B.model_id, envvar="QA_CHAT_MODEL")
@click.option("--num-workers", type=int, default=None, help="Number of chat models to run. Defaults to num GPUs")
@click.option("--dataset", required=True, help="Path to dataset with contexts. Defaults to env QA_DATASET_PATH", envvar="QA_DATASET_PATH")
@click.option(
    "--index",
    help="Path to a pre-built FAISS index over the dataset. Defaults to env QA_INDEX_PATH or <dataset-name>.faiss",
    default=None,
    envvar="QA_INDEX_PATH",
)
def transformers_backend(model_id: str, num_workers: Optional[int], dataset: str, index: Optional[str]) -> QASession:
    os.environ["QA_DATASET_PATH"] = dataset
    if index:
        os.environ["QA_INDEX_PATH"] = index
    args = ["--model-id", model_id]
    if num_workers is not None:
        args.extend(["--num-workers", num_workers])
    run_streamlit("transformer_backend", *args)


@streamlit_cli.command("vllm-client", short_help="Remote vLLM engine")
@click.argument("url")
@click.option("--model-id", help="Chat model ID for prompt formatting.", default=model_configs.VICUNA_7B.model_id, envvar="QA_CHAT_MODEL")
@click.option("--dataset", required=True, help="Path to dataset with contexts. Defaults to env QA_DATASET_PATH", envvar="QA_DATASET_PATH")
@click.option(
    "--index",
    help="Path to a pre-built FAISS index over the dataset. Defaults to env QA_INDEX_PATH or <dataset-name>.faiss",
    default=None,
    envvar="QA_INDEX_PATH",
)
def vllm_client_backend(url: str, model_id: str, dataset: str, index: Optional[str]) -> QASession:
    os.environ["QA_DATASET_PATH"] = dataset
    if index:
        os.environ["QA_INDEX_PATH"] = index
    run_streamlit("vllm_client_backend", url, "--model-id", model_id)


def run_streamlit(backend: str, *args: str):
    streamlit_script_path = os.path.join(settings.PROJECT_ROOT, f"qa/streamlit/{backend}.py")
    sys.argv = ["streamlit", "run", streamlit_script_path, "--", *args]
    runpy.run_module("streamlit", run_name="__main__")


if __name__ == "__main__":
    chat_cli()
