import multiprocessing
from cgitb import text
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Type, Union, Iterable, Any

import torch
from datasets import Dataset, Value
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast, PreTrainedModel
from langchain.vectorstores.base import VectorStore, VST
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings

from bert_qa.data import INDEXED_COLUMN, load_model_datasets, save_data


# Other models to try:
# - context: sentence-transformers/all-mpnet-base-v2
# - context: sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base
#   question: sentence-transformers/facebook-dpr-question_encoder-single-nq-base
# - context: facebook/dpr-ctx_encoder-multiset-base
#   question: facebook/dpr-question_encoder-multiset-base

DEFAULT_CONTEXT_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
# DEFAULT_CONTEXT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
DEFAULT_QUESTION_MODEL = None


@dataclass
class SearchResult:
    source: str
    text: str
    title: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    score: float = 0.0


class Retriever(VectorStore):
    c_tokenizer: PreTrainedTokenizerFast
    c_model_map: Dict[int, PreTrainedModel]
    q_tokenizer: PreTrainedTokenizerFast
    q_model_map: Dict[int, PreTrainedModel]
    
    ### VectorStore methods ###
    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, **kwargs: Any) -> List[str]:
        breakpoint()
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
        kwargs.setdefault("dataset_name", "pubmed")
        search_results = self.search(query, top_k=k, **kwargs)
        return [Document(page_content=result.text, metadata=asdict(result)) for result in search_results]

    @classmethod
    def from_texts(cls: Type[VST], texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]] = None, **kwargs: Any) -> VST:
        breakpoint()
        retriever = cls(load_datasets=False, embedding=embedding, **kwargs)
        retriever.add_texts(texts, metadatas)
        return retriever
    ### End VectorStore ###
    
    def __init__(
        self,
        context_model: str = DEFAULT_CONTEXT_MODEL,
        question_model: Optional[str] = DEFAULT_QUESTION_MODEL,
        model_cls: Union[Type[PreTrainedModel], Type[AutoModel]] = AutoModel,
        tokenizer_cls: Union[Type[PreTrainedTokenizerFast], Type[AutoTokenizer]] = AutoTokenizer,
        load_datasets: bool = True,
        datasets: Optional[Dict[str, Dataset]] = None
    ):
        from multiprocess import set_start_method
        set_start_method('spawn')
        self.context_model = context_model
        self.question_model = question_model
        
        self.c_tokenizer = tokenizer_cls.from_pretrained(self.context_model)
        if question_model:
            self.q_tokenizer = tokenizer_cls.from_pretrained(self.question_model)
        else:
            self.q_tokenizer = self.c_tokenizer
            
        self.c_model_map = None
        self.q_model_map = None
        self.model_cls = model_cls
            
        self.datasets: Dict[str, Dataset] = {}
        if load_datasets:
            self.setup_models()
            for name, dataset in load_model_datasets(self.context_model_name):
                self.datasets[name] = dataset
        if datasets:
            self.datasets.update(datasets)
        
    def setup_models(self):
        if self.c_model_map and self.q_model_map:
            return
        
        if torch.cuda.is_available:
            self.c_model_map = {x: self.model_cls.from_pretrained(self.context_model).to(x) for x in range(torch.cuda.device_count())}
        else:
            self.c_model_map = {0: self.model_cls.from_pretrained(self.context_model).to('cpu')}
            
        if self.question_model:
            if torch.cuda.is_available:
                self.q_model_map = {x: self.model_cls.from_pretrained(self.question_model).to(x) for x in range(torch.cuda.device_count())}
            else:
                self.q_model_map = {0: self.model_cls.from_pretrained(self.question_model).to('cpu')}
        else:
            # Same model/tokenizer for context and question
            self.q_model_map = self.c_model_map
        self.context_model_name = self.c_model_map[0].name_or_path.replace("/", "-").strip("-")
        self.question_model_name = self.q_model_map[0].name_or_path.replace("/", "-").strip("-")

            
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
            meta = {k: context[k] for k in context.keys() if k not in {'source', 'text', 'title'}}
            search_result = SearchResult(context["source"], context["text"], context["title"], score=float(score), meta=meta)
            search_results.append(search_result)
        return search_results

    def split_dataset(self, dataset: Dataset, num_proc=-1) -> Dataset:
        if num_proc == -1:
            num_proc = multiprocessing.cpu_count()
        def _split(batch: Dict[str, List], idx: int) -> Dict[str, List]:
            offset = idx[0]
            text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                self.c_tokenizer,
                chunk_size=256,
                chunk_overlap=20,
                separators=["\n\n", "\n", ".", " "]
            )
            new_rows = []
            text_input = []
            meta = []
            for idx, text in enumerate(batch['text']):
                text_input.append(text)
                meta.append({'index': idx})
            docs = text_splitter.create_documents(text_input, meta)
            columns = batch.keys()
            result = {}
            for column in columns:
                result[column] = [None] * len(docs)
            result['original_index'] = [None] * len(docs)
            for idx, doc in enumerate(docs):
                for column in columns:
                    if column == 'text':
                        result[column][idx] = doc.page_content
                        result['original_index'][idx] = doc.metadata['index'] + offset
                    else:
                        result[column][idx] = batch[column][doc.metadata['index']]
            return result
        return dataset.map(_split, batch_size=100, batched=True, num_proc=num_proc, with_indices=True)


    def add_dataset(self, name: str, dataset: Dataset):
        columns = ["source", "text", "title"]
        if not all([c in dataset.column_names for c in columns]):
            raise Exception("Invalid dataset: Missing required features source, text, and/or title")

        # save_data(dataset, name)
        dataset = self.tokenize_dataset(dataset)
        dataset = self.embed_dataset(dataset)
        dataset.add_faiss_index(INDEXED_COLUMN)
        save_data(dataset, name, self.context_model_name)

        self.datasets[name] = dataset

    def question_embedding(self, text: str, device_num=0) -> torch.Tensor:
        self.setup_models()
        inputs = self.q_tokenizer(
            text, padding=True, truncation=False, verbose=False, return_tensors="pt"
        ).to(device_num)
        with torch.no_grad():
            outputs = self.q_model_map[device_num](inputs.input_ids, attention_mask=inputs.attention_mask)
        return self.cls_pooling(outputs)[0]

    def cls_pooling(self, outputs) -> torch.Tensor:
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]

    def tokenize_dataset(self, dataset: Dataset, max_length: int = 256, overlap_stride: int = 32, batch_size: int = 1000) -> Dataset:
        def _tokenize(batch: Dict[str, List]) -> Dict:
            encoding = self.c_tokenizer(
                batch["text"],
                padding='max_length',
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
        import multiprocessing
        return dataset.map(_tokenize, batched=True, batch_size=batch_size, num_proc=multiprocessing.cpu_count())

    def embed_dataset(self, dataset: Dataset, batch_size: int = 500) -> Dataset:
        self.setup_models()
        import os
        import torch.cuda
        def _embedding(batch: Dict[str, List], rank: int) -> Dict:
            if rank is None:
                rank = 0
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % torch.cuda.device_count())
            with torch.no_grad():
                mask = batch.get("attention_mask")
                input_ids = torch.tensor(batch["input_ids"]).to(rank)
                attention_mask = torch.tensor(mask).to(rank) if mask else None
                outputs = self.c_model_map[rank](input_ids, attention_mask=attention_mask)
            return {
                **batch,
                "embedding": self.cls_pooling(outputs).cpu()
            }
        if len(dataset) == 0:
            return dataset.add_column("embedding", [])
        return dataset.map(_embedding, batched=True, batch_size=batch_size, with_rank=True, num_proc=torch.cuda.device_count())
