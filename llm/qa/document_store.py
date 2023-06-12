from __future__ import annotations
from copy import deepcopy
from datetime import datetime
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
from uuid import uuid4
from time import sleep

from datasets import Dataset
import weaviate
from weaviate.embedded import EmbeddedOptions

from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStoreRetriever, VectorStore
from langchain.vectorstores.weaviate import Weaviate, _default_score_normalizer

from llm.qa.embedding import TextSplitter, QAEmbeddings
from llm.utils.enum import StrEnum

logger = logging.getLogger(__name__)

DEFAULT_INDEX_NAME = "Documentation"


class DataFields(StrEnum):
    """
    Required fields for data in DocStore
    """

    ID = "id"
    TEXT = "text"
    EMBEDDING = "embedding"

    # Split documents
    DOC_ID = "doc_id"
    DOC_OFFSET = "doc_offset"


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
        has_uuids: bool = True,
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

    def parse_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 100,
        source_text_field: Optional[str] = None,
        source_id_field: Optional[str] = None,
        include_meta: Union[bool, str, List[str]] = True,
    ) -> Dataset:
        """
        Parse fields in the given dataset to match the expected format.
        Add UUIDs to each row if not given

        params:
            source_text_field: Text field to exract from the source dataset
            source_id_field: Use custom ID column instead of generating UUIDs
            include_meta: Filter columns that will be kept along with text and ID
        """
        def _parse_batch(
            batch: Dict[str, List],
            source_text_field: str,
            source_id_field: Optional[str] = None,
            include_meta: Optional[List[str]] = None,
        ) -> Dict:
            texts = batch.pop(source_text_field)
            metadata = {}

            if source_id_field or DataFields.ID in batch:
                doc_ids = batch.pop(source_id_field or DataFields.ID)
                doc_ids = [str(uid) for uid in doc_ids]
            else:
                doc_ids = [str(uuid4()) for _ in range(len(texts))]

            if include_meta:
                for key in include_meta:
                    metadata[key] = batch[key]

            return {
                DataFields.ID: doc_ids,
                DataFields.TEXT: texts,
                **metadata,
            }

        if source_text_field is None:
            source_text_field = DataFields.TEXT
        if isinstance(include_meta, bool):
            include_meta = dataset.column_names if include_meta else []
            include_meta = [
                f for f in include_meta if f not in {source_text_field, source_id_field}
            ]
        elif isinstance(include_meta, str):
            include_meta = [include_meta]

        logger.info("Parsing dataset")
        return dataset.map(
            _parse_batch,
            batched=True,
            batch_size=batch_size,
            remove_columns=dataset.column_names,
            fn_kwargs={
                "source_text_field": source_text_field,
                "source_id_field": source_id_field,
                "include_meta": include_meta,
            },
        )

    def split_dataset(
        self,
        dataset: Dataset,
        splitter: TextSplitter,
        batch_size: int = 100,
    ) -> Dataset:
        """
        Split the text in the given dataset into chunks, copying metadata for each split
        and tracking the offset from the parent document.

        Map ID -> DOC_ID, and assign a new ID to each split
        """
        def _split_batch(
            batch: Dict[str, List],
        ) -> Dict:
            texts = batch.pop(DataFields.TEXT)
            split_texts = []
            keys = [DataFields.ID, DataFields.DOC_ID, DataFields.DOC_OFFSET, *batch.keys()]
            split_data: Dict[str, List[Any]] = {key: [] for key in keys}

            for i, text in enumerate(texts):
                splits = splitter.split_text(text)
                split_texts.extend(splits)

                # Duplicate other data for each split, generate IDs, and track offset
                for j in range(len(splits)):
                    split_data[DataFields.ID].append(str(uuid4()))
                    split_data[DataFields.DOC_OFFSET].append(j)
                    for key, vals in batch.items():
                        if key == DataFields.ID:
                            key = DataFields.DOC_ID
                        split_data[key].append(deepcopy(vals[i]))

            return {
                DataFields.TEXT: split_texts,
                **split_data,
            }

        self._validate_dataset(
            dataset, exclude=[DataFields.EMBEDDING, DataFields.DOC_OFFSET, DataFields.DOC_ID]
        )
        logger.info("Splitting dataset")
        dataset = dataset.map(
            _split_batch,
            batched=True,
            batch_size=batch_size,
            remove_columns=dataset.column_names,
        )
        return dataset

    def embed_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 100,
        devices: Optional[Union[str, List[Union[str, int]]]] = None,
    ) -> Dataset:
        """
        Compute embeddings for the text field and add to the dataset

        params:
            devices: List of devices to send the model to for multiprocessing
                Pass "auto" to use all available CUDA devices
        """
        def _embed_batch(batch: Dict[str, List], rank: Optional[int]) -> Dict:
            texts = batch[DataFields.TEXT]
            vectors = embedding_devices[rank or 0].embed_documents(texts)
            return {
                **batch,
                DataFields.EMBEDDING: vectors,
            }

        if devices:
            embedding_devices = self.embedding.multiprocess(devices)
        else:
            embedding_devices = [self.embedding]

        self._validate_dataset(dataset, exclude=[DataFields.EMBEDDING])
        logger.info("Embedding dataset")
        return dataset.map(
            _embed_batch, batched=True, batch_size=batch_size, num_proc=len(embedding_devices), with_rank=True
        )

    def _validate_dataset(self, dataset: Dataset, exclude: Optional[List[str]] = None):
        given = set(dataset.column_names)
        expected = set(DataFields.values())
        if exclude:
            for field in exclude:
                expected.discard(field)

        for field in expected:
            if field not in given:
                raise ValueError(f"Invalid dataset. Expected fields: {expected}. Given: {given}")

    def as_vector_store(self, **kwargs) -> Weaviate:
        kwargs.setdefault("by_text", False)
        return Weaviate(
            self.weaviate_client,
            index_name=self.index_name,
            text_key=DataFields.TEXT,
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
