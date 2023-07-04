import os
from typing import Optional
import click

from llm.qa import model_configs
from llm.qa.embedding import DEFAULT_MODEL, QAEmbeddings
from llm.qa.session import QASession
from llm.qa.vector_store import DatasetVectorStore
from llm.utils.dataset import load_data


@click.command("chat", short_help="Conversational question answering from semantic search")
@click.argument("input-path", required=True, envvar="QA_DATASET_PATH")
@click.option("--input-type", help="Input file type. Defaults to file extension.", default=None, envvar="QA_INPUT_TYPE")
@click.option("--index-path", help="Path to a pre-built FAISS index over the dataset", default=None, envvar="QA_INDEX_PATH")
@click.option("--context-model", help="Model name or path for context embedding", default=DEFAULT_MODEL, envvar="QA_CONTEXT_MODEL")
@click.option("--rephrase", help="Rephrase the question with context from previous messages")
def chat_cli(input_path: str, input_type: Optional[str], index_path: Optional[str], context_model: str, rephrase: bool):
    dataset = load_data(input_path, input_type)
    if index_path is None:
        _index_path = input_path.rsplit(".", 1)[-1] + ".faiss"
        if os.path.isfile(_index_path):
            index_path = _index_path

    model_config = model_configs.VICUNA
    vector_store = DatasetVectorStore(dataset, QAEmbeddings(context_model), index_path=index_path)
    qa_session = QASession.from_model_config(model_config, vector_store)

    while True:
        input_text = input("Question: ")
        qa_session.append_question(input_text)
        question = input_text
        if rephrase:
            question = qa_session.rephrase_question(question)
        qa_session.search_context(question)

        prev_output = ""
        for output_text in qa_session.stream_answer(question):
            new_output = output_text[len(prev_output):]
            prev_output = output_text
            print(new_output, end="", flush=True)
        print()


if __name__ == "__main__":
    chat_cli()
