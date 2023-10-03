from os.path import exists
from typing import List, Set, Optional
from functools import partial

import click
import requests
from bs4 import BeautifulSoup, SoupStrainer
from os.path import join, exists
import gzip
import lxml.etree as etree
import os
import json
import tempfile
import shutil

from tqdm.contrib.concurrent import process_map


@click.group(
    name="pubmed", short_help="Commands for downloading PubMed and formatting it into a dataset"
)
def pubmed_cli():
    pass


@pubmed_cli.command()
@click.argument("output-path")
def download(output_path: str):
    url = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
    resp = requests.get(url)
    links = BeautifulSoup(resp.content, parse_only=SoupStrainer("a"))
    links = list(links)

    from tqdm import tqdm
    import os

    datadir = output_path
    if not exists(datadir):
        os.makedirs(datadir)

    for link in tqdm(links):
        if hasattr(link, "has_attr"):
            if link.has_attr("href"):
                href = link["href"]
                if href.startswith("pubmed23") and href.endswith(".xml.gz"):
                    file_url = f"{url}{href}"
                    target = f"{datadir}/{href}"
                    if exists(target):
                        continue
                    with open(target, "wb+") as f:
                        f.write(requests.get(file_url).content)


@pubmed_cli.command()
@click.argument("input-path")
@click.argument("output-path")
@click.option("--nprocs", default=1)
@click.option("--tmp-path", default="/home/jovyan/tmp")
def jsonl(input_path: str, output_path: str, tmp_path: str, nprocs: int = 1):
    handle_all_xml(input_path, output_path, tmp_path, max_workers=nprocs)


@pubmed_cli.command()
@click.argument("input-path")
@click.argument("output-path")
@click.option("--nprocs", default=1)
@click.option("--titles", "--title", multiple=True, required=False)
def make_single_jsonl(
    input_path: str, output_path: str, nprocs: int = 1, titles: Optional[List[str]] = None
):
    paths = [join(input_path, x) for x in os.listdir(input_path)]
    if not exists(output_path):
        os.makedirs(output_path)
    func = partial(filter_from_file_with_title_search, titles=titles)
    if titles:
        data = process_map(func, paths, max_workers=nprocs)
        with open(join(output_path, "data.jsonl"), "w+") as fout:
            for lines in data:
                for line in lines:
                    fout.write(line)
    else:
        with open(join(output_path, "data.jsonl"), "w+") as fout:
            for path in paths:
                with open(path, "r") as fin:
                    fout.write(fin.read())


@pubmed_cli.command()
@click.argument("input-path")
@click.option("--nprocs", default=1)
def list_titles(input_path: str, nprocs: int = 1):
    paths = [join(input_path, x) for x in os.listdir(input_path)]
    all_titles_sets = process_map(get_titles_from_file, paths, max_workers=nprocs)
    titles_set = set()
    for _ in all_titles_sets:
        titles_set.update(_)
    for title in sorted(titles_set):
        print(title)


def filter_from_file_with_title_search(input_path: str, titles: List[str]) -> List[str]:
    results = []
    with open(input_path, "r") as fin:
        for line in fin:
            data = json.loads(line)
            found = False
            for t in titles:
                if t.lower() in data["publication_title"].lower():
                    found = True
            if found:
                results.append(line)
    return results


def to_string(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return x


def parse_node(node):
    text = None
    source = None
    title = None
    publication_title = None
    _ = node.find(".//Abstract")
    if _ is not None:
        text = etree.tostring(_, encoding="utf-8", method="text")
        text = to_string(text)
    _ = node.find(".//ArticleTitle")
    if _ is not None:
        title = _.text
        title = to_string(title)
    _ = node.find(".//ArticleId[@IdType='doi']")
    if _ is not None:
        source = etree.tostring(_, encoding="utf-8", method="text")
        source = to_string(source)
    journal_title_node = node.find(".//Journal/Title")
    if journal_title_node is not None:
        publication_title = journal_title_node.text
    return dict(text=text, source=source, title=title, publication_title=publication_title)


def handle_one_xml_wrapper(datadir: str, output: str, tmpdir: str, fname: str):
    return handle_one_xml(join(datadir, fname), join(output, fname) + ".jsonl", tmpdir)


def handle_one_xml(input_path: str, output_path: str, tmpdir: str):
    if exists(output_path):
        return
    with gzip.open(input_path) as f:
        tree = etree.parse(f).getroot()
        all_data = []
        for node in tree:
            data = parse_node(node)
            all_data.append(data)
    try:
        with tempfile.NamedTemporaryFile(mode="w+", dir=tmpdir, delete=False) as fout:
            for data in all_data:
                if data["text"] and data["title"]:
                    fout.write(json.dumps(data))
                    fout.write("\n")
            shutil.move(fout.name, output_path)
    except Exception as e:
        raise
    finally:
        if exists(fout.name):
            os.remove(fout.name)


def handle_all_xml(datadir: str, output_path: str, tmpdir: str, max_workers: int) -> None:
    if not exists(output_path):
        os.makedirs(output_path)
    if not exists(tmpdir):
        os.makedirs(tmpdir)

    func = partial(handle_one_xml_wrapper, datadir, output_path, tmpdir)
    if max_workers > 1:
        process_map(func, os.listdir(datadir), max_workers=max_workers)
    else:
        list(map(func, os.listdir(datadir)))


def get_titles_from_file(input_path: str) -> Set[str]:
    titles_set = set()
    with open(input_path, "r") as fin:
        for line in fin:
            data = json.loads(line)
            titles_set.add(data["publication_title"])
    return titles_set


if __name__ == "__main__":
    pubmed_cli()
