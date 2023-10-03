from __future__ import annotations

import logging
from typing import List, Optional, Union

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel
from langchain.embeddings.base import Embeddings

from llm.utils.devices import model_to_devices, parse_device

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
PUBMED_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"


class QAEmbeddings(Embeddings):
    """
    Implements the langchain Embeddings interface with pretrained
    huggingface transformer models.

    For models with only a single set of weights (most of them)
    set only the context model/tokenizer. Questions will be embedded
    using the same model.

    Models that use separate weights to embed contexts and questions,
    (e.g. DPR models) may pass a separate question model/tokenizer.
    """

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
        """
        Copy model onto multiple devices for multiprocessed embeddings
        """
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
            # Some DPR models have their own pooling
            return outputs.pooler_output
        # Default to class pooling. Take the embedding of the CLS token to represent each batch
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
