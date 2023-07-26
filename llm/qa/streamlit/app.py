import os
from typing import List, Optional

import streamlit as st
from langchain.vectorstores.base import VectorStore

from llm.qa import model_configs
from llm.qa.embedding import DEFAULT_MODEL, QAEmbeddings
from llm.qa.inference import TransformersEngine, InferenceEngine, MultiprocessEngine
from llm.qa.session import QASession
from llm.qa.parser import DataFields
from llm.qa.vector_store import DatasetVectorStore
from llm.utils.dataset import load_data

QA_DATASET_PATH = os.environ["QA_DATASET_PATH"]
QA_INDEX_PATH = os.getenv("QA_INDEX_PATH")
QA_CONTEXT_MODEL = os.getenv("QA_CONTEXT_MODEL", DEFAULT_MODEL)
model_config = model_configs.VICUNA

MARKDOWN_LINEBREAK = "  \n"


@st.cache_resource
def get_inference_engine(num_workers: Optional[int] = None) -> InferenceEngine:
    # MultiprocessEngine ensures sessions gets dedicated access to a model
    # while their request is being processed. By default, one inference engine will
    # be loaded to each available GPU device.
    return MultiprocessEngine(_load_engine, num_workers=num_workers)


def _load_engine(local_rank: int) -> TransformersEngine:
    # Loads model to the device specified by rank, and initialized the inference engine
    return TransformersEngine.from_model_config(model_config, device=local_rank)


@st.cache_resource
def get_vector_store(dataset_path: str, index_path: Optional[str] = None) -> DatasetVectorStore:
    # VectorStore for semantic search. Shared by all sessions
    dataset = load_data(dataset_path)
    embedding = QAEmbeddings(QA_CONTEXT_MODEL)
    return DatasetVectorStore(dataset, embedding, index_path=index_path)


def get_qa_session(engine: InferenceEngine, vector_store: VectorStore) -> QASession:
    # Conversation/contexts for each session
    if "qa_session" not in st.session_state:
        qa_session = QASession.from_model_config(
            model_config, vector_store, engine=engine, debug=True
        )
        st.session_state["qa_session"] = qa_session
        return qa_session
    return st.session_state["qa_session"]


def filter_contexts():
    # Filter which contexts are seen by the LLM for the next question
    contexts = []
    for doc, to_include in zip(qa_session.results, included):
        if to_include:
            contexts.append(doc.page_content)

    qa_session.set_contexts(contexts)


# Ensure only the main proc interacts with streamlit
if __name__ == "__main__":
    st.set_page_config(page_title="QA Chat", page_icon=":robot_face:", layout="wide")

    engine = get_inference_engine()
    vector_store = get_vector_store(QA_DATASET_PATH)
    qa_session = get_qa_session(engine, vector_store)
    output = st.text("")
    included: List[bool] = []

    clear_convo = st.button("clear conversation")
    if clear_convo:
        # Clear conversation, but keep system prompt in case the
        # user wants to re-query over the previous context.
        qa_session.clear(keep_results=True)

    with st.form(key="input_form", clear_on_submit=True):
        # Collect user input
        user_input = st.text_area("You:", key="input", height=100)
        placeholder = st.container()
        query_submitted = st.form_submit_button(label="Query")

        rephrase_question = placeholder.checkbox(
            "Rephrase question with history",
            key="rephrase_question",
            value=True,
            disabled=not (query_submitted or qa_session.has_history),
        )
        search_new_context = placeholder.checkbox(
            "Search new context",
            key="search_new_context",
            value=True,
            disabled=not (query_submitted or qa_session.results),
        )

        if query_submitted and not user_input:
            query_submitted = False

    question = ""
    if query_submitted and not clear_convo:
        # Write user input out to streamlit, then search for contexts
        qa_session.append_question(user_input)
        output.write(qa_session.get_history(separator=MARKDOWN_LINEBREAK))

        with st.spinner("Searching..."):
            if rephrase_question:
                question = qa_session.rephrase_question(user_input)
            else:
                question = user_input
            if search_new_context:
                qa_session.search_context(question)


    if qa_session.results:
        # Write contexts out to streamlit, with checkboxes to filter what is sent to the LLM
        with st.form(key="checklists"):
            for i, doc in enumerate(qa_session.results):
                include = st.checkbox("include in chat context", key=i, value=True)
                included.append(include)
                st.write(doc.page_content)
                st.write({k: v for k, v in doc.metadata.items() if k != DataFields.EMBEDDING})
                st.divider()

            checklist_submitted = st.form_submit_button(label="Filter")
            if checklist_submitted:
                filter_contexts()

    if not clear_convo:
        if query_submitted:
            # Stream response from LLM, updating chat window at each step
            history = qa_session.get_history(separator=MARKDOWN_LINEBREAK, range_start=0) + MARKDOWN_LINEBREAK
            for text in qa_session.stream_answer(question, with_prefix=True):
                message = history + text
                output.write(message)
        else:
            # Write existing message history
            message = qa_session.get_history(separator=MARKDOWN_LINEBREAK)
            output.write(message)
