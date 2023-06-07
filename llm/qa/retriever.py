from curses import meta
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any, Dict, Iterable, List, Optional, Type, Union
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings

import torch
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast, PreTrainedModel
from langchain.vectorstores.base import VectorStore, VST
from langchain.text_splitter import RecursiveCharacterTextSplitter as lRCTS

from llm.qa.data import INDEXED_COLUMN, load_model_datasets, save_data
from llm.qa.embedding import RecursiveCharacterTextSplitter

# Other models to try:
# - context: sentence-transformers/all-mpnet-base-v2
# - context: sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base
#   question: sentence-transformers/facebook-dpr-question_encoder-single-nq-base
# - context: facebook/dpr-ctx_encoder-multiset-base
#   question: facebook/dpr-question_encoder-multiset-base

DEFAULT_CONTEXT_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
DEFAULT_QUESTION_MODEL = None


@dataclass
class Content:
    source: str
    text: str
    title: Optional[str] = None


@dataclass
class SearchResult(Content):
    score: float = 0.0


class Retriever(VectorStore):
    c_tokenizer: PreTrainedTokenizerFast
    c_model: PreTrainedModel
    q_tokenizer: PreTrainedTokenizerFast
    q_model: PreTrainedModel

    def __init__(
        self,
        context_model: str = DEFAULT_CONTEXT_MODEL,
        question_model: Optional[str] = DEFAULT_QUESTION_MODEL,
        model_cls: Union[Type[PreTrainedModel], Type[AutoModel]] = AutoModel,
        tokenizer_cls: Union[Type[PreTrainedTokenizerFast], Type[AutoTokenizer]] = AutoTokenizer,
        load_datasets: bool = True,
        embedding: Optional[Embeddings] = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.c_tokenizer = tokenizer_cls.from_pretrained(context_model)
        self.c_model = model_cls.from_pretrained(context_model).to(self.device)
        if question_model:
            self.q_tokenizer = tokenizer_cls.from_pretrained(question_model)
            self.q_model = model_cls.from_pretrained(question_model).to(self.device)
        else:
            # Same model/tokenizer for context and question
            self.q_tokenizer = self.c_tokenizer
            self.q_model = self.c_model

        self.vector_store: Optional[Dataset] = None
        self.embedding = embedding

        self.datasets: Dict[str, Dataset] = {}
        if load_datasets:
            for name, dataset in load_model_datasets(self.context_model_name):
                self.datasets[name] = dataset

    ### VectorStore methods ###
    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, **kwargs: Any) -> List[str]:
        metadatas = metadatas or [{}] * len(texts)
        dataset = Dataset.from_dict({"text": texts, "metadata": metadatas})
        dataset = self.tokenize_dataset(dataset)
        dataset = self.embed_dataset(dataset)

        if self.vector_store:
            for item in dataset:
                self.vector_store.add_item(item)
        else:
            self.vector_store = dataset

    def similarity_search(self, query: str, k: int = 3, **kwargs: Any) -> List[Document]:
        kwargs.setdefault("dataset_name", "kubernetes")
        search_results = self.search(query, top_k=k, **kwargs)
        return [Document(page_content=result.text, metadata=asdict(result)) for result in search_results]

    @classmethod
    def from_texts(cls: Type[VST], texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]] = None, **kwargs: Any) -> VST:
        retriever = cls(load_datasets=False, embedding=embedding, **kwargs)
        retriever.add_texts(texts, metadatas)
        return retriever
    ### End VectorStore ###

    @property
    def context_model_name(self) -> str:
        return self.c_model.name_or_path.replace("/", "-").strip("-")

    @property
    def question_model_name(self) -> str:
        return self.q_model.name_or_path.replace("/", "-").strip("-")

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
        embedding = self.question_embedding(question)
        results = dataset.search(INDEXED_COLUMN, embedding.cpu().numpy(), k=top_k)
        search_results = []
        for i, score in zip(results.indices, results.scores):
            context = dataset[int(i)]
            search_result = SearchResult(context["source"], context["text"], context["title"], float(score))
            search_results.append(search_result)
        return search_results

    def add_dataset(self, name: str, dataset: Dataset):
        columns = ["source", "text", "title"]
        if not all([c in dataset.column_names for c in columns]):
            raise Exception("Invalid dataset: Missing required features source, text, and/or title")

        dataset = self.tokenize_dataset(dataset)
        dataset = self.embed_dataset(dataset)
        dataset.add_faiss_index(INDEXED_COLUMN)
        self.datasets[name] = dataset
        return dataset

    def save_dataset(self, name: str):
        dataset = self.datasets.get(name)
        if not dataset:
            raise Exception(f"Dataset {name} not found")
        save_data(dataset, name, self.context_model_name)

    def question_embedding(self, text: str) -> torch.Tensor:
        inputs = self.q_tokenizer(
            text, padding=True, truncation=False, verbose=False, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.q_model(inputs.input_ids, attention_mask=inputs.attention_mask)
        return self.cls_pooling(outputs)[0]

    def cls_pooling(self, outputs) -> torch.Tensor:
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]

    def tokenize_dataset(self, dataset: Dataset, max_length: int = 256, overlap_stride: int = 32, batch_size: int = 1000) -> Dataset:
        def _tokenize(batch: Dict[str, List]) -> Dict:
            encoding = self.c_tokenizer(
                batch["text"],
                padding=True,
                truncation=True,
                max_length=max_length,
                stride=overlap_stride,
                return_overflowing_tokens=True,
                return_attention_mask=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )

            # There may be more output rows than input rows. Map data in the old rows to the new rows.
            overflow_mapping = encoding.overflow_to_sample_mapping
            offset_mapping = encoding.offset_mapping
            batch_overflowed: Dict[str, List] = {}
            for key, values in batch.items():
                if key == "text":
                    # Extract original text by overflow and offset mappings
                    new_val = [""] * len(overflow_mapping)
                    for i, j in enumerate(overflow_mapping):
                        offsets = offset_mapping[i]
                        non_zero = torch.nonzero(offsets)
                        if len(non_zero) > 0:
                            first_nonzero = int(non_zero[0][0])
                            last_nonzero = int(non_zero[-1][0])
                            start = offsets[first_nonzero][0]
                            end = offsets[last_nonzero][-1]
                            new_val[i] = values[j][start:end]
                        else:
                            # No tokens in text (probably empty)
                            new_val[i] = values[j]
                else:
                    new_val = [None] * len(overflow_mapping)
                    for i, j in enumerate(overflow_mapping):
                        new_val[i] = values[j]
                batch_overflowed[key] = new_val

            return {
                **batch_overflowed,
                "input_ids": encoding.input_ids,
                "attention_mask": encoding.attention_mask,
                "offset_mapping": offset_mapping,
            }

        if len(dataset) == 0:
            dataset = dataset.add_column("input_ids", [])
            dataset = dataset.add_column("attention_mask", [])
            return dataset.add_column("offset_mapping", [])
        return dataset.map(_tokenize, batched=True, batch_size=batch_size)

    def embed_dataset(self, dataset: Dataset, batch_size: int = 100) -> Dataset:
        def _embedding(batch: Dict[str, List]) -> Dict:
            with torch.no_grad():
                mask = batch.get("attention_mask")
                input_ids = torch.tensor(batch["input_ids"]).to(self.device)
                attention_mask = torch.tensor(mask).to(self.device) if mask else None
                outputs = self.c_model(input_ids, attention_mask=attention_mask)
            return {
                **batch,
                "embedding": self.cls_pooling(outputs).cpu()
            }
        if len(dataset) == 0:
            return dataset.add_column("embedding", [])
        return dataset.map(_embedding, batched=True, batch_size=batch_size)
