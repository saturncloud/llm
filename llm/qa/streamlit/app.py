import os
from typing import List, Optional
import streamlit as st
from langchain.vectorstores.base import VectorStore

from llm.inference.base import InferenceEngine
from llm.model_configs import VICUNA_7B, ModelConfig
from llm.qa.embedding import DEFAULT_MODEL, QAEmbeddings
from llm.qa.parser import DataFields
from llm.qa.session import QASession
from llm.qa.vector_store import DatasetVectorStore
from llm.utils.data import load_data

QA_DATASET_PATH = os.environ["QA_DATASET_PATH"]
QA_INDEX_PATH = os.getenv("QA_INDEX_PATH")
QA_CONTEXT_MODEL = os.getenv("QA_CONTEXT_MODEL", DEFAULT_MODEL)
QA_CHAT_MODEL = os.getenv("QA_CHAT_MODEL", VICUNA_7B.model_id)


@st.cache_resource
def get_vector_store(dataset_path: Optional[str] = None, index_path: Optional[str] = None) -> DatasetVectorStore:
    if not dataset_path:
        dataset_path = QA_DATASET_PATH
    if not index_path:
        index_path = QA_INDEX_PATH

    # VectorStore for semantic search. Shared by all sessions
    dataset = load_data(dataset_path)
    embedding = QAEmbeddings(QA_CONTEXT_MODEL)
    return DatasetVectorStore(dataset, embedding, index_path=index_path)


def get_qa_session(model_config: ModelConfig, engine: InferenceEngine, vector_store: VectorStore, debug: bool = True, **kwargs) -> QASession:
    # Conversation/contexts for each session
    if "qa_session" not in st.session_state:
        qa_session = QASession.from_model_config(
            model_config, vector_store, engine=engine, debug=debug, **kwargs
        )
        st.session_state["qa_session"] = qa_session
        return qa_session
    return st.session_state["qa_session"]


def render_app(qa_session: QASession):
    chat_container = st.container()
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
        num_contexts = st.number_input(label="Num Contexts", min_value=0, value=3)
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

    if query_submitted and not clear_convo:
        # Write user input out to streamlit, then search for contexts
        qa_session.append_question(user_input)
        with chat_container:
            for message in qa_session.conversation.messages:
                with st.chat_message("user"):
                    st.write(message.input)

                if message.response is not None:
                    with st.chat_message("assistant"):
                        st.write(message.response)

        with st.spinner("Searching..."):
            if rephrase_question:
                search_query = qa_session.rephrase_question(user_input)
            else:
                search_query = user_input
            if search_new_context:
                qa_session.search_context(search_query, top_k=num_contexts)


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
                filter_contexts(qa_session, included)

    if not clear_convo:
        if query_submitted:
            # Stream response from LLM, updating chat window at each step
            with chat_container, st.chat_message("assistant"):
                answer = st.text("")
                # TODO: Handle labeling here and remove it from qa_session
                for text in qa_session.stream_answer(user_input):
                    answer.write(text)


def filter_contexts(qa_session: QASession, included: List[bool]):
    # Filter which contexts are seen by the LLM for the next question
    contexts = []
    for doc, to_include in zip(qa_session.results, included):
        if to_include:
            contexts.append(doc.page_content)

    qa_session.set_contexts(contexts)
