from argparse import ArgumentParser
import os
from typing import Dict, Optional
from urllib.parse import urlparse
import streamlit as st
import torch

from llm.inference import InferenceEngine, MultiprocessEngine, VLLMClient
from llm.model_configs import ModelConfig, VicunaConfig, bnb_quantization
from llm.qa.embedding import DEFAULT_EMBEDDING_MODEL

DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL_ID", VicunaConfig.model_id)
DEFAULT_CONTEXT_MODEL = os.getenv("QA_CONTEXT_MODEL", DEFAULT_EMBEDDING_MODEL)
DEFAULT_DATASET_PATH = os.getenv("QA_DATASET_PATH")
DEFAULT_INDEX_PATH = os.getenv("QA_INDEX_PATH")


def setup_page(title: str):
    st.set_page_config(page_title=title, page_icon=":robot_face:", layout="wide")
    st.title(title, anchor=False)
    style_text()


def get_engine() -> InferenceEngine:
    if "engine" in st.session_state:
        return st.session_state["engine"]
    init_session_state()

    backend = st.session_state.get("backend", "transformers")
    engine_kwargs = st.session_state.get("engine_kwargs", {})

    if backend == "transformers":
        engine = get_transformers_engine(**engine_kwargs)
    elif backend == "vllm-client":
        engine = get_vllm_client_engine(**engine_kwargs)
    else:
        raise Exception(f'Unknown engine type "{backend}"')
    st.session_state["engine"] = engine
    return engine


def init_session_state():
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--qa-dataset-path", help="Document QA dataset path", default=DEFAULT_DATASET_PATH
    )
    parser.add_argument(
        "-i", "--qa-index-path", help="Document QA FAISS index path", default=DEFAULT_INDEX_PATH
    )
    parser.add_argument(
        "-c",
        "--qa-context-model",
        help="Document QA embedding model for vector search",
        default=DEFAULT_CONTEXT_MODEL,
    )

    subparsers = parser.add_subparsers(help="backend", dest="backend", required=False)
    transformers_parser = subparsers.add_parser("transformers", help="Local transformers engine")
    transformers_parser.add_argument("-m", "--model-id", help="Chat model ID", default=DEFAULT_MODEL_ID)
    transformers_parser.add_argument(
        "-n", "--num-workers", help="Number of chat models to run. Defaults to num GPUs."
    )
    vllm_client_parser = subparsers.add_parser("vllm-client", help="Remote VLLM engine")
    vllm_client_parser.add_argument("--url", required=True, help="Base URL for vLLM API")
    vllm_client_parser.add_argument(
        "-m",
        "--model-id",
        help="Chat model ID for determining prompt format",
        default=DEFAULT_MODEL_ID,
    )
    args = parser.parse_args()

    if not args.backend:
        args.backend = "transformers"
        args.model_id = DEFAULT_MODEL_ID
        args.num_workers = None

    model_config = ModelConfig.from_registry(args.model_id)
    st.session_state["model_config"] = model_config

    # Token generation settings
    st.session_state.setdefault("temperature", 0.7)
    st.session_state.setdefault("top_p", 0.9)
    if model_config.max_length > 2048:
        st.session_state.setdefault("max_new_tokens", 512)
    else:
        st.session_state.setdefault("max_new_tokens", 256)

    # Docstore settings
    st.session_state["qa_dataset_path"] = args.qa_dataset_path
    st.session_state["qa_index_path"] = args.qa_index_path
    st.session_state["qa_context_model"] = args.qa_context_model

    # Backend settings
    st.session_state["backend"] = args.backend
    if args.backend == "transformers":
        st.session_state["engine_kwargs"] = {
            "model_config": model_config,
            "num_workers": args.num_workers
        }
    else:  # vllm-client
        st.session_state["engine_kwargs"] = {"url": args.url}


@st.cache_resource
def get_transformers_engine(
    model_config: ModelConfig, num_workers: Optional[int] = None
) -> MultiprocessEngine:
    # MultiprocessEngine ensures sessions gets dedicated access to a model
    # while their request is being processed. By default, one inference engine will
    # be loaded to each available GPU device.
    return MultiprocessEngine.from_model_config(
        model_config,
        num_workers=num_workers,
        model_kwargs={
            "torch_dtype": torch.float16,
            "quantization_config": bnb_quantization(),
        },
    )


def get_vllm_client_engine(url: str) -> VLLMClient:
    return VLLMClient(url, headers=saturn_headers(url))


def generation_settings():
    """
    Shared token generation settings

    Managing session_state explicitly to persist across pages, streamlit widgets with
    keys get cleared from session_state.
    """
    max_new_tokens = st.number_input(
        label="Max New Tokens", min_value=1, value=st.session_state.get("max_new_tokens", 256)
    )
    temperature = st.slider(
        label="Temperature",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=st.session_state.get("temperature", 0.7),
    )
    top_p = st.slider(
        label="Top P",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=st.session_state.get("top_p", 0.9),
    )

    st.session_state.max_new_tokens = max_new_tokens
    st.session_state.temperature = temperature
    st.session_state.top_p = top_p
    return {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p}


def saturn_headers(url: str) -> Dict[str, str]:
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


def chat_bubble(role: str, text: str = ""):
    with st.chat_message(role):
        text_box = st.text("")
        text_box.text(text)
    return text_box


def style_text():
    st.markdown(
        """
        <style>
            div[data-testid="stText"] {
                text-wrap: wrap;
                word-break: break-word;
                font-family: Consolas, "Liberation Mono", Courier, monospace;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
