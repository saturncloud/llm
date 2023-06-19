from __future__ import annotations
from copy import deepcopy
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
from time import sleep

from datasets import Dataset
import weaviate

from langchain.vectorstores.base import VectorStoreRetriever
from langchain.vectorstores.weaviate import Weaviate

from llm.qa.embedding import TextSplitter, QAEmbeddings
from llm.qa.parser import DataFields

logger = logging.getLogger(__name__)

DEFAULT_INDEX_NAME = "Documentation"


class DocStore:
    def __init__(
        self,
        embedding: QAEmbeddings,
        weaviate_client: Optional[weaviate.Client] = None,
        index_name: str = DEFAULT_INDEX_NAME,
    ) -> None:
        """
        Manage weaviate vector/doc store with custom embeddings
        """
        if weaviate_client is None:
            weaviate_client = init_weaviate()
        self.weaviate_client = weaviate_client
        self.index_name = index_name
        self.embedding = embedding

    def add_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 1000,
        has_uuids: Optional[bool] = None,
    ):
        object_ids = []
        def _add_batch(batch: Dict[str, List]) -> Dict:
            embeddings = batch.pop(DataFields.EMBEDDING)
            uuids = batch.pop(DataFields.ID) if has_uuids else None
            data_objects = to_row_format(batch)

            ids = self._add_embedded(
                data_objects=data_objects,
                embeddings=embeddings,
                uuids=uuids,
            )
            object_ids.append(ids)

        if has_uuids is None:
            has_uuids = DataFields.ID.value in dataset.column_names

        self._validate_dataset(dataset)
        dataset.map(_add_batch, batched=True, batch_size=batch_size)
        return object_ids

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
        uuids: Optional[List[str]] = None,
    ) -> List[str]:
        if embeddings is None:
            embeddings = self.embedding.embed_documents(texts)
        data_objects = [{} for _ in range(len(texts))]
        for i, text in enumerate(texts):
            data_objects[i][DataFields.TEXT] = text
            if metadatas:
                data_objects[i].update(metadatas[i])

        return self._add_embedded(data_objects, embeddings, uuids=uuids)

    def _add_embedded(
        self,
        data_objects: List[Dict[str, Any]],
        embeddings: List[List[float]],
        uuids: Optional[List[str]] = None,
    ) -> List[str]:
        object_ids = []
        with self.weaviate_client.batch as batch:
            for i, (data_object, embedding) in enumerate(zip(data_objects, embeddings)):
                object_id = batch.add_data_object(
                    data_object=data_object,
                    class_name=self.index_name,
                    uuid=uuids[i] if uuids else None,
                    vector=embedding,
                )
                object_ids.append(object_id)
        return object_ids

    def search(
        self,
        question: str,
        top_k: int = 3,
        include_fields: Optional[List[str]] = None,
        include_additional: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        fields = set([str(DataFields.TEXT)])
        additional = set([str(DataFields.ID), "distance"])
        if include_fields:
            fields.update(include_fields)
        if include_additional:
            additional.update(include_additional)

        vector = self.embedding.embed_query(question)
        results = (
            self.weaviate_client.query
            .get(self.index_name, list(fields))
            .with_additional(list(additional))
            .with_near_vector({"vector": vector})
            .with_limit(top_k)
            .do()
        )
        if "data" not in results:
            raise Exception(results)

        data_objects = results["data"]["Get"][self.index_name]
        for data in data_objects:
            _additional = data.pop("_additional")
            data.update(_additional)

        return data_objects

    def as_vector_store(self, **kwargs) -> Weaviate:
        kwargs.setdefault("by_text", False)
        return Weaviate(
            self.weaviate_client,
            index_name=self.index_name,
            text_key=str(DataFields.TEXT),
            embedding=self.embedding,
            **kwargs,
        )

    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        return self.as_vector_store().as_retriever(**kwargs)


def init_weaviate(wait_timeout: int = 60, **kwargs) -> weaviate.Client:
    if "url" not in kwargs and "embedded_options" not in kwargs:
        # Use local "embedded" (poor name choice) weaviate running in subproc
        kwargs["embedded_options"] = weaviate.EmbeddedOptions()
    weaviate.Config()
    client = weaviate.Client(**kwargs)

    if wait_timeout > 0:
        start = datetime.utcnow()
        while not client.is_ready():
            sleep(1)
            delta = datetime.utcnow() - start
            if delta.total_seconds() > wait_timeout:
                raise TimeoutError(
                    f"Timed out waiting for weaviate to be ready after {wait_timeout} seconds"
                )
    return client


def to_row_format(batch: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    row_formatted = []
    for key, vals in batch.items():
        for i, val in enumerate(vals):
            if len(row_formatted) < i + 1:
                row_formatted.append({})
            row_formatted[i][key] = val
    return row_formatted
