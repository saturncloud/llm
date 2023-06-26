from math import e
from typing import List, Optional
from urllib.parse import urlparse

import click
from transformers import AutoTokenizer

from llm.qa.crawler import DocSpider
from llm.qa.embedding import DEFAULT_MODEL, QAEmbeddings, RecursiveCharacterTextSplitter
from llm.qa.parser import DatasetParser, DataFields
from llm.utils.cli import click_coroutine
from llm.utils.dataset import load_data, save_data


@click.group(name="data", short_help="Commands for parsing datasets to be used in semantic search")
def data_cli():
    pass


@data_cli.command(short_help="Scrape the given URL to a document dataset")
@click_coroutine
@click.argument("url", required=True)
@click.option("-o", "--output-path", required=True, help="Output path for the dataset")
@click.option("domains", "-d", "--domain", multiple=True, help="One or more domains that can be scraped from the url. Any by default.", default=None)
@click.option("--link-regex", help="Match URLs against regex before adding to scrape pipeline", default=None)
@click.option("--text-css", help="Only extract text from elements matching the given CSS selector", default=None)
@click.option("--max-depth", default=10, type=int, help="Maximum depth of URL links to follow")
async def scrape(
    url: str,
    output_path: str,
    domains: List[str],
    link_regex: Optional[str],
    text_css: Optional[str],
    max_depth: int,
):
    parsed = urlparse(url)
    if parsed.hostname is None:
        raise Exception(f"Invalid URL: {url}")

    dataset = DocSpider.run(
        url,
        allowed_domains=domains,
        link_regex=link_regex,
        text_css=text_css,
        settings={"DEPTH_LIMIT": max_depth, "CONCURRENT_REQUESTS": 20, "LOG_LEVEL": "INFO"},
    )
    save_data(dataset, output_path)


@data_cli.command(short_help="Format text and ID fields of the dataset to known keys, generating a UUID for each row if needed")
@click.argument("input-path", required=True)
@click.option("-o", "--output-path", required=True, help="Output path for the parsed dataset")
@click.option("--input-type", help="Input file type. Defaults to file extension.")
@click.option("--batch-size", help="Batch size for processing rows of the dataset", default=100)
@click.option("--source-text-field", help="Text field of the source dataset", default=str(DataFields.TEXT))
@click.option("--source-id-field", help="ID field of the source dataset", default=None)
def format(
    input_path: str,
    output_path: str,
    input_type: Optional[str],
    batch_size: int,
    source_text_field: str,
    source_id_field: Optional[str],
):
    dataset = load_data(input_path, input_type)
    parser = DatasetParser()

    dataset = parser.format(
        dataset,
        batch_size=batch_size,
        source_text_field=source_text_field,
        source_id_field=source_id_field,
    )
    save_data(dataset, output_path)


@data_cli.command(short_help="Split the dataset on its text field such that each chunk is of a given maximum size")
@click.argument("input-path", required=True)
@click.option("-o", "--output-path", required=True, help="Output path for the parsed dataset")
@click.option("--input-type", help="Input file type. Defaults to file extension.")
@click.option("--batch-size", help="Batch size for processing rows of the dataset", default=100)
@click.option("--chunk-size", help="Max chunk size of final text in number of tokens", default=256)
@click.option("--chunk-overlap", help="Number of tokens to overlap between chunks", default=32)
@click.option("--context-model", help="Model name or path for splitting contexts by token length", default=DEFAULT_MODEL, envvar="QA_CONTEXT_MODEL")
def split(
    input_path: str,
    output_path: str,
    input_type: Optional[str],
    batch_size: int,
    chunk_size: int,
    chunk_overlap: int,
    context_model: str,
):
    dataset = load_data(input_path, input_type)
    tokenizer = AutoTokenizer.from_pretrained(context_model)
    parser = DatasetParser()
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    dataset = parser.split(dataset, splitter, batch_size=batch_size)
    save_data(dataset, output_path)


@data_cli.command(short_help="Embed the text field of the dataset for semantic search, and save the result")
@click.argument("input-path", required=True)
@click.option("-o", "--output-path", required=True, help="Output path for the parsed dataset")
@click.option("--input-type", help="Input file type. Defaults to file extension.")
@click.option("--batch-size", help="Batch size for processing rows of the dataset", default=100)
@click.option("devices", "-d", "--device", multiple=True, help="One or more devices to run embeddings on. Pass 'auto' to auto-detect multiple-gpus.", default=None)
@click.option("--context-model", help="Model name or path for context embedding", default=DEFAULT_MODEL, envvar="QA_CONTEXT_MODEL")
def embed(
    input_path: str,
    output_path: str,
    input_type: Optional[str],
    batch_size: int,
    devices: Optional[List[str]],
    context_model: str,
):
    dataset = load_data(input_path, input_type)
    embedding = QAEmbeddings(context_model)
    embedding_devices = embedding.multiprocess(*devices) if devices else [embedding]
    parser = DatasetParser(*embedding_devices)

    dataset = parser.embed(dataset, batch_size=batch_size)
    save_data(dataset, output_path)


@data_cli.command(short_help="Full processing pipeline for getting a document dataset ready to be indexed")
@click.argument("input-path", required=True)
@click.option("-o", "--output-path", required=True, help="Output path for the parsed dataset")
@click.option("--input-type", help="Input file type. Defaults to file extension.")
@click.option("--batch-size", help="Batch size for processing rows of the dataset", default=100)
@click.option("devices", "-d", "--device", multiple=True, help="One or more devices to run embeddings on. Pass 'auto' to auto-detect multiple-gpus.", default=None)
@click.option("--context-model", help="Model name or path for context embedding", default=DEFAULT_MODEL, envvar="QA_CONTEXT_MODEL")
def pipeline(input_path: str, output_path: str, input_type: Optional[str], batch_size: int, devices: Optional[List[str]], context_model: str):
    dataset = load_data(input_path, input_type)
    embedding = QAEmbeddings(context_model)
    embedding_devices = embedding.multiprocess(*devices) if devices else [embedding]
    parser = DatasetParser(*embedding_devices)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        embedding.context_tokenizer,
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=256,
        chunk_overlap=32,
    )

    dataset = parser.format(dataset, batch_size=batch_size)
    dataset = parser.split(dataset, splitter, batch_size=batch_size)
    dataset = parser.embed(dataset, batch_size=batch_size)
    save_data(dataset, output_path)


@data_cli.command(short_help="Index the dataset's embedding column with FAISS")
@click.argument("input-path", required=True)
@click.option("-o", "--output-path", required=True, help="Output path for the FAISS index")
@click.option("--input-type", help="Input file type. Defaults to file extension.")
@click.option("--index-name", help="Name of the index to add the dataset to", default=None)
def index(input_path: str, output_path: str, input_type: Optional[str], index_name: Optional[str]):
    dataset = load_data(input_path, input_type)
    parser = DatasetParser()
    kwargs = {}
    if index_name:
        kwargs["index_name"] = index_name
    parser.index(dataset, index_path=output_path, **kwargs)



if __name__ == "__main__":
    data_cli()
