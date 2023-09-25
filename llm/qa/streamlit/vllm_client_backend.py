from argparse import ArgumentParser
import os
from typing import Dict
from urllib.parse import urlparse

import streamlit as st

from llm.inference import VLLMClient
from llm.model_configs import ChatModelConfig, ModelConfig
from llm.qa.prompts import FEW_SHOT
from llm.qa.streamlit.app import QA_CHAT_MODEL, get_qa_session, get_vector_store, render_app


def get_inference_engine(url: str) -> VLLMClient:
    # Load one client per session
    if "engine" not in st.session_state:
        engine = VLLMClient(url, headers=headers(url))
        st.session_state["engine"] = engine
    return st.session_state["engine"]


def headers(url: str) -> Dict[str, str]:
    # Check if we need to set Saturn auth header
    headers = {}
    SATURN_TOKEN = os.getenv("SATURN_TOKEN")
    SATURN_BASE_URL = os.getenv("SATURN_BASE_URL")
    if SATURN_TOKEN and SATURN_BASE_URL:
        saturn_base = urlparse(SATURN_BASE_URL).hostname
        saturn_base = saturn_base.split(".", 1)[-1]
        _url = urlparse(url)
        if _url.scheme == "https" and _url.hostname and _url.hostname.endswith(saturn_base):
            headers["Authorization"] = f"token {SATURN_TOKEN}"
    return headers


# Ensure only the main proc interacts with streamlit
if __name__ == "__main__":
    st.set_page_config(page_title="QA Chat", page_icon=":robot_face:", layout="wide")

    parser = ArgumentParser()
    parser.add_argument("url", help="Base URL for vLLM API")
    parser.add_argument(
        "-m", "--model-id", help="Chat model ID for determining prompt format", default=QA_CHAT_MODEL
    )
    args = parser.parse_args()

    model_config = ChatModelConfig.from_registry(args.model_id)
    engine = VLLMClient(args.url, headers=headers(args.url))
    vector_store = get_vector_store()
    qa_session = get_qa_session(model_config, engine, vector_store, prompt=FEW_SHOT)

    render_app(qa_session)
