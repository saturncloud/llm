from argparse import ArgumentParser
from typing import Optional

import streamlit as st
import torch

from llm.inference import InferenceEngine, MultiprocessEngine
from llm.model_configs import ModelConfig, bnb_quantization
from llm.qa.streamlit.app import QA_CHAT_MODEL, get_qa_session, get_vector_store, render_app


@st.cache_resource
def get_inference_engine(
    model_config: ModelConfig, num_workers: Optional[int] = None
) -> InferenceEngine:
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


# Ensure only the main proc interacts with streamlit
if __name__ == "__main__":
    st.set_page_config(page_title="QA Chat", page_icon=":robot_face:", layout="wide")

    parser = ArgumentParser()
    parser.add_argument("-m", "--model-id", help="Chat model ID", default=QA_CHAT_MODEL)
    parser.add_argument(
        "-n", "--num-workers", help="Number of chat models to run. Defaults to num GPUs."
    )
    args = parser.parse_args()

    model_config = ModelConfig.from_registry(args.model_id)
    engine = get_inference_engine(model_config, num_workers=args.num_workers)

    vector_store = get_vector_store()
    qa_session = get_qa_session(model_config, engine, vector_store, debug=True)

    kwargs = {}
    if model_config.max_length > 2048:
        # Expanded defaults for long context models
        kwargs["max_new_tokens"] = 512
    render_app(qa_session, **kwargs)
