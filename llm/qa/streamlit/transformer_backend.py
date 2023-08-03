from argparse import ArgumentParser
import os
from typing import Dict, Optional
from urllib.parse import urlparse
import click

import streamlit as st
from langchain.vectorstores.base import VectorStore

from llm.inference import InferenceEngine, MultiprocessEngine, VLLMClient
from llm.model_configs import VICUNA_7B, ModelConfig
from llm.qa.embedding import DEFAULT_MODEL, QAEmbeddings
from llm.qa.session import QASession
from llm.qa.streamlit.app import render_app
from llm.qa.vector_store import DatasetVectorStore
from llm.utils.data import load_data

QA_DATASET_PATH = os.environ["QA_DATASET_PATH"]
QA_INDEX_PATH = os.getenv("QA_INDEX_PATH")
QA_CONTEXT_MODEL = os.getenv("QA_CONTEXT_MODEL", DEFAULT_MODEL)
QA_CHAT_MODEL = os.getenv("QA_CHAT_MODEL", VICUNA_7B.model_id)

MARKDOWN_LINEBREAK = "  \n"


@st.cache_resource
def get_inference_engine(model_config: ModelConfig, num_workers: Optional[int] = None) -> InferenceEngine:
    # MultiprocessEngine ensures sessions gets dedicated access to a model
    # while their request is being processed. By default, one inference engine will
    # be loaded to each available GPU device.
    return MultiprocessEngine.from_model_config(model_config, num_workers=num_workers)


@st.cache_resource
def get_vector_store(dataset_path: str, index_path: Optional[str] = None) -> DatasetVectorStore:
    # VectorStore for semantic search. Shared by all sessions
    dataset = load_data(dataset_path)
    embedding = QAEmbeddings(QA_CONTEXT_MODEL)
    return DatasetVectorStore(dataset, embedding, index_path=index_path)


def get_qa_session(model_config: ModelConfig, engine: InferenceEngine, vector_store: VectorStore, **kwargs) -> QASession:
    # Conversation/contexts for each session
    if "qa_session" not in st.session_state:
        qa_session = QASession.from_model_config(
            model_config, vector_store, engine=engine, **kwargs
        )
        st.session_state["qa_session"] = qa_session
        return qa_session
    return st.session_state["qa_session"]


def headers(url: str) -> Dict[str, str]:
    headers = {}
    SATURN_TOKEN = os.getenv("SATURN_TOKEN")
    SATURN_BASE_URL = os.getenv("SATURN_BASE_URL")
    if SATURN_TOKEN and SATURN_BASE_URL:
        # Check if we need to set Saturn auth header
        saturn_base = urlparse(SATURN_BASE_URL)
        _url = urlparse(url)
        if _url.scheme == "https" and _url.hostname and _url.hostname.endswith(saturn_base.hostname):
            headers["Authorization"] = f"token {SATURN_TOKEN}"
    return headers


# Ensure only the main proc interacts with streamlit
if __name__ == "__main__":
    st.set_page_config(page_title="QA Chat", page_icon=":robot_face:", layout="wide")

    parser = ArgumentParser()
    parser.add_argument("-m", "--model-id", help="Chat model ID", default=QA_CHAT_MODEL)
    parser.add_argument("-n", "--num-workers", help="Number of chat models to run. Defaults to num GPUs.")
    args = parser.parse_args()

    model_config = ModelConfig.from_registry(args.model_id)
    engine = get_inference_engine(model_config, num_workers=args.num_workers)

    vector_store = get_vector_store(QA_DATASET_PATH)
    qa_session = get_qa_session(engine, vector_store, model_config=model_config)

    render_app(qa_session)
