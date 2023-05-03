from dataclasses import asdict, dataclass, field
import os
from typing import Any, Dict, List, Optional
from numpy import indices

import torch
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

CONTEXTS_DIR = os.getenv("CONTEXTS_DIR", "/data/contexts")
DATA_FILE = "data.jsonl"
INDEX_FILE = "index.faiss"
INDEX_NAME = "embedding"


@dataclass
class Content:
    source: str
    text: str
    title: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContextChunk(Content):
    embedding: List[float] = field(default_factory=list)


@dataclass
class SearchResult(Content):
    score: float = 0.0


class Retriever:
    def __init__(self, model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"):
        self.context = []
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: PreTrainedModel = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float32
        )
        self.datasets: Dict[str, Dataset] = {}
        for dir in os.listdir(CONTEXTS_DIR):
            self.load_dataset(dir)

    def search(self, question: str, dataset_name: Optional[str] = None, top_k: int = 5) -> List[SearchResult]:
        if dataset_name:
            return self._search(question, dataset_name, top_k=top_k)

        all_results: List[SearchResult] = []
        for name in self.datasets.keys():
            results = self._search(question, name, top_k=top_k)
            all_results.extend(results)
        all_results = sorted(all_results, key=lambda r: r.score)
        if len(all_results) <= top_k:
            return all_results
        return all_results[:top_k]

    def _search(self, question: str, dataset_name: str, top_k: int = 5) -> List[SearchResult]:
        dataset = self.datasets[dataset_name]
        encoding = self.cls_embedding(question)
        results = dataset.search(INDEX_NAME, encoding.numpy(), k=top_k)
        search_results = []
        for i, score in zip(results.indices, results.scores):
            context = dataset[int(i)]
            search_result = SearchResult(context["source"], context["text"], context["title"], float(score))
            search_results.append(search_result)
        return search_results


    def load_dataset(self,  name: str) -> Dataset:
        data_path = self.data_path(name)
        index_path = self.index_path(name)
        dataset = Dataset.from_json(data_path)
        if os.path.isfile(index_path):
            dataset.load_faiss_index(INDEX_NAME, index_path)
        else:
            dataset.add_faiss_index(INDEX_NAME)
            dataset.save_faiss_index(INDEX_NAME, index_path)
        self.datasets[name] = dataset
        return dataset

    def add_dataset(self, dataset_name: str, content_list: List[Content]) -> Dataset:
        chunks = []
        for content in content_list:
            contexts = self.get_contexts(content)
            chunks.extend([asdict(c) for c in contexts])

        dataset = Dataset.from_list(chunks)
        data_path = self.data_path(dataset_name)
        index_path = self.index_path(dataset_name)
        dataset.to_json(data_path)
        dataset.add_faiss_index(INDEX_NAME)
        dataset.save_faiss_index(INDEX_NAME, index_path)
        self.datasets[dataset_name] = dataset
        return dataset

    def get_contexts(self, content: Content) -> List[ContextChunk]:
        split = self.split_context(content.text)
        contexts: List[ContextChunk] = []
        for text_chunk in split:
            embedding = self.cls_embedding(text_chunk)
            contexts.append(ContextChunk(
                text=text_chunk,
                source=content.source,
                title=content.title,
                embedding=embedding.tolist()
            ))
        return contexts

    def split_context(self, context: str, max_length: int = 384, overlap: int = 64) -> List[str]:
        def _token_length(text: str) -> int:
            return len(self.tokenizer.encode(text, padding=False, truncation=False, verbose=False))

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=max_length,
            chunk_overlap=overlap,
            length_function=_token_length,
        )
        return splitter.split_text(context)

    def cls_embedding(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            encoding = self.model(inputs.input_ids)
        return encoding.last_hidden_state[0, 0]

    def data_path(self, name: str) -> str:
        return os.path.join(CONTEXTS_DIR, name, DATA_FILE)

    def index_path(self, name: str) -> str:
        return os.path.join(CONTEXTS_DIR, name, INDEX_FILE)
