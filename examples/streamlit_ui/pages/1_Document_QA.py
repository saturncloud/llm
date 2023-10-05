import os
from typing import List, Optional
import streamlit as st
from langchain.vectorstores.base import VectorStore

from llm.inference.base import InferenceEngine
from llm.model_configs import ModelConfig, VicunaConfig
from llm.qa.embedding import DEFAULT_MODEL, QAEmbeddings
from llm.qa.parser import DataFields
from llm.qa.session import QASession
from llm.qa.vector_store import DatasetVectorStore
from llm.utils.data import load_data

from examples.streamlit_ui.components import chat_bubble, generation_settings, get_engine, setup_page

QA_DATASET_PATH = os.environ["QA_DATASET_PATH"]
QA_INDEX_PATH = os.getenv("QA_INDEX_PATH")
QA_CONTEXT_MODEL = os.getenv("QA_CONTEXT_MODEL", DEFAULT_MODEL)
QA_CHAT_MODEL = os.getenv("QA_CHAT_MODEL", VicunaConfig.model_id)


@st.cache_resource
def get_vector_store(
    dataset_path: Optional[str] = None, index_path: Optional[str] = None
) -> DatasetVectorStore:
    if not dataset_path:
        dataset_path = QA_DATASET_PATH
    if not index_path:
        index_path = QA_INDEX_PATH

    # VectorStore for semantic search. Shared by all sessions
    dataset = load_data(dataset_path)
    embedding = QAEmbeddings(QA_CONTEXT_MODEL)
    return DatasetVectorStore(dataset, embedding, index_path=index_path)


def get_qa_session(
    model_config: ModelConfig,
    engine: InferenceEngine,
    vector_store: VectorStore,
    debug: bool = True,
    **kwargs
) -> QASession:
    # Conversation/contexts for each session
    if "qa_session" not in st.session_state:
        qa_session = QASession.from_model_config(
            model_config, vector_store, engine=engine, debug=debug, **kwargs
        )
        st.session_state.qa_session = qa_session
        return qa_session
    return st.session_state.qa_session


def filter_contexts(qa_session: QASession, included: List[bool]):
    # Filter which contexts are seen by the LLM for the next question
    contexts = []
    for doc, to_include in zip(qa_session.results, included):
        if to_include:
            contexts.append(doc.page_content)

    qa_session.set_contexts(contexts)


if __name__ == "__main__":
    setup_page("Document QA")

    engine = get_engine()
    vector_store = get_vector_store()
    qa_session = get_qa_session(st.session_state.model_config, engine, vector_store)

    chat_container = st.container()
    clear_convo = st.button("Clear")
    if clear_convo:
        # Clear conversation, but keep system prompt in case the
        # user wants to re-query over the previous context.
        qa_session.clear(keep_results=True)

    with st.form(key="input_form", clear_on_submit=True):
        # Collect user input
        user_input = st.text_area("Query:", key="input", height=100)
        contexts_only = st.checkbox(label="Contexts only", value=False)
        query_submitted = st.form_submit_button(label="Submit")
        if clear_convo or (query_submitted and not user_input):
            query_submitted = False

    with st.sidebar:
        st.header("Settings")
        with st.form(key="settings_form", clear_on_submit=False):
            rephrase_question = st.checkbox(
                "Rephrase context query with history",
                key="rephrase_question",
                value=True,
            )
            search_new_context = st.checkbox(
                "Search new contexts on chat",
                key="search_new_context",
                value=True,
            )
            num_contexts = st.number_input(label="Num Contexts", min_value=0, value=3)
            generation_kwargs = generation_settings()
            st.form_submit_button(label="Apply")

    # Write current chat
    if query_submitted and not contexts_only:
        qa_session.append_question(user_input)

    with chat_container:
        for message in qa_session.conversation.messages:
            chat_bubble("user", message.input)
            if message.response is not None:
                chat_bubble("assistant", message.response)

    if query_submitted and (search_new_context or contexts_only):
        # Rephrase and search for contexts
        with st.spinner("Searching..."):
            if rephrase_question:
                search_query = qa_session.rephrase_question(
                    user_input,
                    max_new_tokens=256,
                    temperature=generation_kwargs["temperature"],
                    top_p=generation_kwargs["top_p"],
                )
            else:
                search_query = user_input
            st.session_state.search_query = search_query
            qa_session.search_context(search_query, top_k=num_contexts)

    # Write contexts out to streamlit, with checkboxes to filter what is sent to the LLM
    included: List[bool] = []
    with st.sidebar:
        st.header("Contexts")
        if "search_query" in st.session_state:
            st.caption(st.session_state.search_query)
        with st.form(key="checklists"):
            for i, doc in enumerate(qa_session.results):
                include = st.checkbox(
                    "include in chat context", key=i, value=True, disabled=search_new_context
                )
                included.append(include)
                st.write(doc.page_content)
                st.json(
                    {k: v for k, v in doc.metadata.items() if k != DataFields.EMBEDDING},
                    expanded=False,
                )
                st.divider()

            checklist_submitted = st.form_submit_button(
                label="Filter", disabled=(not qa_session.results)
            )
            if checklist_submitted:
                filter_contexts(qa_session, included)

    if query_submitted and not contexts_only:
        # Stream response from LLM, updating chat window at each step
        with chat_container:
            answer = chat_bubble("assistant")
            for text in qa_session.stream_answer(user_input, **generation_kwargs):
                answer.write(text)
