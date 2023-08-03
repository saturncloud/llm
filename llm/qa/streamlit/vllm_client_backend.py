from argparse import ArgumentParser
import os
from typing import Dict, Optional
from urllib.parse import urlparse

import streamlit as st
from langchain.vectorstores.base import VectorStore

from llm.inference import VLLMClient
from llm.model_configs import VICUNA_7B, ModelConfig
from llm.qa.embedding import DEFAULT_MODEL, QAEmbeddings
from llm.qa.prompts import FEW_SHOT
from llm.qa.session import QASession
from llm.qa.streamlit.app import render_app
from llm.qa.vector_store import DatasetVectorStore
from llm.utils.data import load_data

QA_DATASET_PATH = os.environ["QA_DATASET_PATH"]
QA_INDEX_PATH = os.getenv("QA_INDEX_PATH")
QA_CONTEXT_MODEL = os.getenv("QA_CONTEXT_MODEL", DEFAULT_MODEL)
QA_CHAT_MODEL = os.getenv("QA_CHAT_MODEL", VICUNA_7B.model_id)

MARKDOWN_LINEBREAK = "  \n"


def get_inference_engine(url: str) -> VLLMClient:
    # Load one client per session
    if "engine" not in st.session_state:
        engine = VLLMClient(url, headers=headers(url))
        st.session_state["engine"] = engine
    return st.session_state["engine"]


@st.cache_resource
def get_vector_store(dataset_path: str, index_path: Optional[str] = None) -> DatasetVectorStore:
    # VectorStore for semantic search. Shared by all sessions
    dataset = load_data(dataset_path)
    embedding = QAEmbeddings(QA_CONTEXT_MODEL)
    return DatasetVectorStore(dataset, embedding, index_path=index_path)


def get_qa_session(engine: VLLMClient, vector_store: VectorStore, **kwargs) -> QASession:
    # Conversation/contexts for each session
    if "qa_session" not in st.session_state:
        qa_session = QASession(engine, vector_store, **kwargs)
        st.session_state["qa_session"] = qa_session
    return st.session_state["qa_session"]


def headers(url: str) -> Dict[str, str]:
    # Check if we need to set Saturn auth header
    headers = {}
    SATURN_TOKEN = os.getenv("SATURN_TOKEN")
    SATURN_BASE_URL = os.getenv("SATURN_BASE_URL")
    if SATURN_TOKEN and SATURN_BASE_URL:
        saturn_base = urlparse(SATURN_BASE_URL)
        _url = urlparse(url)
        if _url.scheme == "https" and _url.hostname and _url.hostname.endswith(saturn_base.hostname):
            headers["Authorization"] = f"token {SATURN_TOKEN}"
    return headers


# Ensure only the main proc interacts with streamlit
if __name__ == "__main__":
    st.set_page_config(page_title="QA Chat", page_icon=":robot_face:", layout="wide")

    parser = ArgumentParser()
    parser.add_argument("url", help="Base URL for vLLM API")
    args = parser.parse_args()

    engine = VLLMClient(args.url, headers=headers(args.url))
    vector_store = get_vector_store(QA_DATASET_PATH)
    qa_session = get_qa_session(engine, vector_store, prompt=FEW_SHOT)

    render_app(qa_session)
