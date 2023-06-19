from __future__ import annotations

from typing import Any, Iterable, List, Optional
from datasets import Dataset

import numpy as np

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

from llm.qa.parser import DEFAULT_INDEX_NAME, DataFields, DatasetParser


class DatasetVectorStore(VectorStore):
    def __init__(
        self,
        dataset: Dataset,
        embedding: Embeddings,
        index_name: str = DEFAULT_INDEX_NAME,
        index_path: Optional[str] = None,
        **kwargs,
    ):
        self.dataset = dataset
        self.embedding = embedding
        self.index_name = index_name
        if index_path:
            self.dataset.load_faiss_index(index_name, index_path, **kwargs)
        elif not self.dataset.is_index_initialized(self.index_name):
            self.dataset.add_faiss_index(str(DataFields.EMBEDDING), self.index_name, **kwargs)

    def save_index(self, index_path: str, **kwargs):
        self.dataset.save_faiss_index(self.index_name, index_path, **kwargs)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        uuids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        parser = DatasetParser(self.embedding)
        dataset = parser.create_dataset(texts, metadatas=metadatas, uuids=uuids, **kwargs)
        self.add_dataset(dataset)

        return [row[DataFields.ID] for row in dataset]

    def add_dataset(self, dataset: Dataset):
        if set(dataset.column_names) != set(self.dataset.column_names):
            raise Exception(f"Invalid columns in dataset. Expected {self.dataset.column_names}")

        for row in dataset:
            self.dataset.add_item(row)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        query_embedding = self.embedding.embed_query(query)
        np_embedding = np.asarray(query_embedding, dtype=np.float32)
        results = self.dataset.search(self.index_name, np_embedding, k, **kwargs)

        documents: List[Document] = []
        score_attr = "score" if "score" not in self.dataset.column_names else "_score"
        for i, score in zip(results.indices, results.scores):
            row = self.dataset[int(i)]
            text = row[DataFields.TEXT]
            metadata = {k: v for k, v in row.items() if k != DataFields.TEXT}
            metadata[score_attr] = score
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        uuids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> DatasetVectorStore:
        parser = DatasetParser(embedding)
        dataset = parser.create_dataset(texts, metadatas=metadatas, uuids=uuids, **kwargs)
        return cls(dataset, embedding, **kwargs)
