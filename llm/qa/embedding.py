from __future__ import annotations
import copy
from abc import ABC, abstractmethod

import logging
from typing import Any, Callable, Iterable, List, Optional, Sequence, Union

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import BaseDocumentTransformer

from llm.utils.devices import model_to_devices, parse_device

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"


class QAEmbeddings(Embeddings):
    context_model: PreTrainedModel
    context_tokenizer: PreTrainedTokenizerBase
    question_model: PreTrainedModel
    question_tokenizer: PreTrainedTokenizerBase

    def __init__(
        self,
        context_model: Union[str, PreTrainedModel] = DEFAULT_MODEL,
        context_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        question_model: Optional[Union[str, PreTrainedModel]] = None,
        question_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        device: Union[str, int, None] = None,
    ):
        if device is None:
            if isinstance(context_model, PreTrainedModel):
                device = context_model.device
            elif isinstance(question_model, PreTrainedModel):
                device = question_model.device
        device = parse_device(device)

        # Load context model/tokenizer
        if isinstance(context_model, str):
            context_model = AutoModel.from_pretrained(context_model).to(device)
        self.context_model = context_model

        if context_tokenizer is None:
            context_tokenizer = AutoTokenizer.from_pretrained(context_model.name_or_path)
        self.context_tokenizer = context_tokenizer

        # Load question model/tokenizer if set, else same as context model
        if isinstance(question_model, str):
            question_model = AutoModel.from_pretrained(question_model).to(device)
            if question_tokenizer is None:
                question_tokenizer = AutoTokenizer.from_pretrained(question_model.name_or_path)
        self.question_model = question_model or self.context_model
        self.question_tokenizer = question_tokenizer or self.context_tokenizer

    def multiprocess(
        self,
        *devices: Union[str, int],
        set_start_method: Optional[bool] = None,
    ) -> List[QAEmbeddings]:
        embeddings = []
        if "auto" in devices:
            # Select all available devices
            devices = []
        c_models = model_to_devices(self.context_model, *devices)
        if self.question_model == self.context_model:
            q_models = c_models
        else:
            q_models = model_to_devices(self.question_model, *devices)

        for c_model, q_model in zip(c_models, q_models):
            embeddings.append(
                QAEmbeddings(
                    context_model=c_model,
                    question_model=q_model,
                    context_tokenizer=self.context_tokenizer,
                    question_tokenizer=self.question_tokenizer,
                )
            )

        if set_start_method is None:
            set_start_method = len(embeddings) > 1 and torch.cuda.is_available()
        if set_start_method:
            # Required to fork a process using CUDA
            import multiprocess
            multiprocess.set_start_method("spawn")
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], is_context=False).tolist()[0]

    def _embed(self, texts: List[str], is_context: bool = True) -> torch.Tensor:
        model = self.context_model if is_context else self.question_model
        tokenizer = self.context_tokenizer if is_context else self.question_tokenizer
        inputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
            verbose=False,
        ).to(model.device)
        with torch.no_grad():
            outputs = model(inputs.input_ids, attention_mask=inputs.attention_mask)
        return self.sentence_pooling(outputs)

    def sentence_pooling(self, outputs) -> torch.Tensor:
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]

    def batch_token_length(self, texts: List[str], is_context: bool = True) -> List[int]:
        tokenizer = self.context_tokenizer if is_context else self.question_tokenizer
        return tokenizer(
            texts,
            truncation=False,
            padding=False,
            verbose=False,
            return_length=True,
        ).lengths


class TextSplitter(BaseDocumentTransformer, ABC):
    """Interface for splitting text into chunks."""

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        batched_length_function: Optional[Callable[[List[str]], List[int]]] = None,
    ):
        """Create a new TextSplitter."""
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        if batched_length_function is None:
            batched_length_function = lambda texts: [self._length_function(t) for t in texts]
        self._batched_length_function = batched_length_function

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            for chunk in self.split_text(text):
                new_doc = Document(
                    page_content=chunk, metadata=copy.deepcopy(_metadatas[i])
                )
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(self, splits: Iterable[str], separator: str, lengths: Optional[Iterable[int]] = None) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._batched_length_function([separator])[0]

        docs = []
        current_doc: List[str] = []
        current_doc_lengths: List[int] = []
        total = 0
        split_lengths = lengths or self._batched_length_function(splits)
        for d, _len in zip(splits, split_lengths):
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= current_doc_lengths[0] + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
                        current_doc_lengths = current_doc_lengths[1:]
            current_doc.append(d)
            current_doc_lengths.append(_len)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    @classmethod
    def from_huggingface_tokenizer(cls, tokenizer: Any, **kwargs: Any) -> TextSplitter:
        """Text splitter that uses HuggingFace tokenizer to count length."""
        try:
            from transformers import PreTrainedTokenizerBase

            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise ValueError(
                    "Tokenizer received was not an instance of PreTrainedTokenizerBase"
                )
            # Ignore warning about token length
            tokenizer.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True

            def _huggingface_tokenizer_length(texts: List[str]) -> List[int]:
                if not texts:
                    return []
                return tokenizer(texts, truncation=False, return_length=True).length

        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )
        return cls(batched_length_function=_huggingface_tokenizer_length, **kwargs)

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform sequence of documents by splitting them."""
        return self.split_documents(list(documents))

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Asynchronously transform a sequence of documents by splitting them."""
        raise NotImplementedError


class RecursiveCharacterTextSplitter(TextSplitter):
    """Implementation of splitting text that looks at characters.

    Recursively tries to split by different characters to find one
    that works.
    """

    def __init__(self, separators: Optional[List[str]] = None, **kwargs: Any):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = self._separators[-1]
        for _s in self._separators:
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                break
        # Now that we have the separator, split the text
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _good_split_lengths = []
        split_lengths = self._batched_length_function(splits)
        for s, _len in zip(splits, split_lengths):
            if _len < self._chunk_size:
                _good_splits.append(s)
                _good_split_lengths.append(_len)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, separator, lengths=_good_split_lengths)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                    _good_split_lengths = []
                other_info = self.split_text(s)
                final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, separator, lengths=_good_split_lengths)
            final_chunks.extend(merged_text)
        return final_chunks
