from os.path import exists
from typing import List, Set, Optional
from functools import partial

import click
from datasets import Dataset
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup, SoupStrainer, element
from os.path import join, exists
import gzip
import lxml.etree as etree
from tqdm import tqdm
import os
import json
import tempfile
import shutil

from tqdm.contrib.concurrent import process_map

from bert_qa.fastchatter import chat_loop
from bert_qa.retriever import Retriever
from bert_qa.data import INDEXED_COLUMN


model = "pritamdeka/S-PubMedBert-MS-MARCO"


@click.group()
def cli():
    pass

@cli.command()
@click.argument('path')
def download(path: str):
    url = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
    resp = requests.get(url)
    links = BeautifulSoup(resp.content, parse_only=SoupStrainer('a'))
    links = list(links)

    from tqdm import tqdm
    import os

    datadir = path
    if not exists(datadir):
        os.makedirs(datadir)

    for link in tqdm(links):
        if hasattr(link, 'has_attr'):
            if link.has_attr('href'):
                href = link['href']
                if href.startswith('pubmed23') and href.endswith('.xml.gz'):
                    file_url = f"{url}{href}"
                    target = f"{datadir}/{href}"
                    if exists(target):
                        continue
                    with open(target, "wb+") as f:
                        f.write(requests.get(file_url).content)

def to_string(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return x


def parse(node):
    text = None
    source = None
    title = None
    _ = node.find('.//AbstractText')
    if _ is not None:
        text = _.text
        text = to_string(text)
    _ = node.find('.//ArticleTitle')
    if _ is not None:
        title = _.text
        title = to_string(title)
    _ = node.find('.//ArticleTitle')
    if _ is not None:
        source = etree.tostring(_).decode('utf-8')
        source = to_string(source)
    journal_title_node = node.find('.//Journal/Title')
    if journal_title_node is not None:
        publication_title = journal_title_node.text
    return dict(text=text, source=source, title=title, publication_title=publication_title)


def handle_one_xml(input_path: str, output_path: str, tmpdir: str):
    if exists(output_path):
        return
    with gzip.open(input_path) as f:
        tree = etree.parse(f).getroot()
        all_data = []
        for node in tree:
            data = parse(node)
            all_data.append(data)
    try:
        with tempfile.NamedTemporaryFile(mode='w+', dir=tmpdir, delete=False) as fout:
            for data in all_data:
                if data['text'] and data['title']:
                    fout.write(json.dumps(data))
                    fout.write('\n')
            shutil.move(fout.name, output_path)
    except Exception as e:
        raise
    finally:
        if exists(fout.name):
            os.remove(fout.name)


def handle_all_xml(datadir: str, output: str, tmpdir: str, max_workers: int) -> None:
    if not exists(output):
        os.makedirs(output)
    if not exists(tmpdir):
        os.makedirs(tmpdir)
    def partial(fname):
        return handle_one_xml(join(datadir, fname), join(output, fname) + ".jsonl", tmpdir)
    process_map(partial, os.listdir(datadir), max_workers=max_workers)


@cli.command()
@click.argument('inpath')
@click.argument('outpath')
@click.option('--nprocs', default=1)
@click.option('--tmppath', default="/home/jovyan/tmp")
def jsonl(inpath: str, outpath: str, tmppath: str, nprocs: int = 1):
    handle_all_xml(inpath, outpath, tmppath, max_workers=nprocs)


def filter_from_file_with_title_search(inpath: str, titles: List[str]) -> List[str]:
    results = []
    with open(inpath, "r") as fin:
        for line in fin:
            data = json.loads(line)
            found = False
            for t in titles:
                if t.lower() in data['publication_title'].lower():
                    found = True
            if found:
                results.append(line)
    return results


@cli.command()
@click.argument('inpath')
@click.argument('outpath')
@click.option('--nprocs', default=1)
@click.option("--titles", "--title", multiple=True, required=False)
def make_single_jsonl(inpath: str, outpath: str, nprocs: int = 1, titles: Optional[List[str]] = None):
    paths = [join(inpath, x) for x in os.listdir(inpath)]
    if not exists(outpath):
        os.makedirs(outpath)
    func = partial(filter_from_file_with_title_search, titles=titles)
    if titles:
        data = process_map(func, paths, max_workers=nprocs)
        with open(join(outpath, "data.jsonl"), "w+") as fout:
            for lines in data:
                for line in lines:
                    fout.write(line)
    else:
        with open(join(outpath, "data.jsonl"), "w+") as fout:
            for path in paths:
                with open(path, "r") as fin:
                    fout.write(fin.read())


def get_titles_from_file(inpath: str) -> Set[str]:
    titles_set = set()
    with open(inpath, "r") as fin:
        for line in fin:
            data = json.loads(line)
            titles_set.add(data['publication_title'])
    return titles_set


@cli.command()
@click.argument('inpath')
@click.option('--nprocs', default=1)
def list_titles(inpath: str, nprocs: int = 1):
    paths = [join(inpath, x) for x in os.listdir(inpath)]
    all_titles_sets = process_map(get_titles_from_file, paths, max_workers=nprocs)
    titles_set = set()
    for _ in all_titles_sets:
        titles_set.update(_)
    for title in sorted(titles_set):
        print(title)


@cli.command()
@click.argument('inpath')
@click.argument('outpath')
def download(path: str):
    url = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
    resp = requests.get(url)
    links = BeautifulSoup(resp.content, parse_only=SoupStrainer('a'))
    links = list(links)

    from tqdm import tqdm
    import os

    datadir = path
    if not exists(datadir):
        os.makedirs(datadir)

    for link in tqdm(links):
        if hasattr(link, 'has_attr'):
            if link.has_attr('href'):
                href = link['href']
                if href.startswith('pubmed23') and href.endswith('.xml.gz'):
                    file_url = f"{url}{href}"
                    target = f"{datadir}/{href}"
                    if exists(target):
                        continue
                    with open(target, "wb+") as f:
                        f.write(requests.get(file_url).content)

@cli.command()
@click.argument('inpath')
@click.argument('outpath')
def split(inpath: str, outpath:str):
    ds = Dataset.from_json(inpath)
    retriever = Retriever(context_model=model, question_model=model, load_datasets=False)
    ds = retriever.split_dataset(ds)
    ds.to_parquet(outpath)


@cli.command()
@click.argument('inpath')
@click.argument('outpath')
def tokenize(inpath: str, outpath:str):
    ds = Dataset.from_parquet(inpath)
    retriever = Retriever(context_model=model, question_model=model, load_datasets=False)
    ds = retriever.tokenize_dataset(ds)
    ds.to_parquet(outpath)


@cli.command()
@click.argument('inpath')
@click.argument('outpath')
@click.argument('indexpath')
def embed(inpath: str, outpath: str, indexpath: str):
    ds = Dataset.from_parquet(inpath)
    retriever = Retriever(context_model=model, question_model=model, load_datasets=False)
    ds = retriever.embed_dataset(ds)
    ds.add_faiss_index(INDEXED_COLUMN)
    ds.save_faiss_index(INDEXED_COLUMN, indexpath)
    ds.to_parquet(outpath)


@cli.command()
@click.argument('inpath')
@click.argument('indexpath')
@click.argument('query')
def retrieve(inpath: str, indexpath: str, query: str):
    ds = Dataset.from_parquet(inpath)
    ds.load_faiss_index(INDEXED_COLUMN, indexpath)
    retriever = Retriever(context_model=model, question_model=model, load_datasets=False, datasets={'pubmed': ds})
    result = retriever.search(query)
    print(result)


@cli.command()
def chat():
    chat_loop()


if __name__ == "__main__":
    cli()