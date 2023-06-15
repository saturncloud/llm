import click

from llm.qa import model_configs
from llm.qa.embedding import QAEmbeddings
from llm.qa.fastchatter import FastchatEngine, QASession


@click.command("chat", short_help="Conversational question answering from semantic search")
@click.argument("index-name")
def chat_cli(index_name: str):
    from llm.qa.document_store import DocStore

    model_config = model_configs.VICUNA
    model, tokenizer = model_config.load()
    engine = FastchatEngine(model, tokenizer, model_config.max_length)
    docstore = DocStore(QAEmbeddings(), index_name=index_name)
    qa_session = QASession.from_model_config(model_config, engine, docstore)

    while True:
        input_text = input("Question: ")
        qa_session.update_context(input_text)
        qa_session.append_question(input_text)
        prev_output = ""
        for output_text in qa_session.conversation_stream():
            new_output = output_text[len(prev_output):]
            prev_output = output_text
            print(new_output, end="", flush=True)
        print()


if __name__ == "__main__":
    chat_cli()
