import os
import torch
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from datasets import Dataset
from transformers.pipelines import TextGenerationPipeline

from langchain.chains import LLMChain, RetrievalQA, RetrievalQAWithSourcesChain

from bert_qa.retriever import Retriever

from langchain.llms import HuggingFacePipeline
from transformers import pipeline, LlamaForCausalLM, LlamaTokenizer

from langchain.docstore.document import Document
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from typing import Dict, List, Optional, Type, Union, Iterable, Any
import torch

from transformers import pipeline, LlamaForCausalLM, LlamaTokenizer, TextGenerationPipeline
from langchain.chains import LLMChain, RetrievalQA, RetrievalQAWithSourcesChain
from langchain.llms import HuggingFacePipeline

from bert_qa.retriever import Retriever
from bert_qa.data import INDEXED_COLUMN


model = "pritamdeka/S-PubMedBert-MS-MARCO"
ds = Dataset.from_parquet("/home/jovyan/output-cardiology/pubmed/data-embeddings.parquet")
ds.load_faiss_index(INDEXED_COLUMN, "/home/jovyan/output-cardiology/pubmed/data-embeddings.faiss")
retriever = Retriever(context_model=model, question_model=model, load_datasets=False, datasets={'pubmed': ds})


base_model = "/home/jovyan/workspace/models/vicuna-7b"
load_in_8bit = True
use_cuda = True
if torch.cuda.is_available() and use_cuda:
    device = "cuda"
    device_map = "auto"
    low_cpu_mem_usage = None
else:
    device = "cpu"
    device_map = {"": device}
    low_cpu_mem_usage = True

model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=load_in_8bit,
    torch_dtype=torch.float16,
    device_map=device_map,
    low_cpu_mem_usage=low_cpu_mem_usage,
)
tokenizer = LlamaTokenizer.from_pretrained(base_model)

model.eval()
cls_name = model.__class__.__name__
if torch.__version__ >= "2":
    model = torch.compile(model)
    model.__class__.__name__ = cls_name

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    # top_p=0.95,
    repetition_penalty=1.0,
)
llm = HuggingFacePipeline(pipeline=pipe)

# retriever = Retriever()

# More simple retriever
# qa = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff", retriever=retriever.as_retriever())

class RetrievalQAWithOptions(RetrievalQA):
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]

        docs = self._get_docs(question)
        self.combine_documents_chain.run
        results = self.combine_documents_chain(
            dict(input_documents=docs, question=question), callbacks=_run_manager.get_child()
        )
        results[self.output_key] = results.pop(self.combine_documents_chain.output_key, "")

        if self.return_source_documents:
            return {**results, "source_documents": docs}
        else:
            return results

        
qa = RetrievalQAWithOptions.from_chain_type(llm, chain_type="map_reduce", retriever=retriever.as_retriever(), chain_type_kwargs={"return_map_steps": True})
while True:
    result = qa({qa.input_keys[0]: input("\nUser: ")}, return_only_outputs=True)
    answer = result.pop(qa.output_key)
    for key, val in result.items():
        print(f"{key}:\n-----------------------------\n{val}\n")

    print(f"Final Answer: {answer}")
