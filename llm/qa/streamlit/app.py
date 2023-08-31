from typing import List
import streamlit as st

from llm.qa.parser import DataFields
from llm.qa.session import QASession

MARKDOWN_LINEBREAK = "  \n"


def render_app(qa_session: QASession):
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
                filter_contexts(qa_session, included)

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


def filter_contexts(qa_session: QASession, included: List[bool]):
    # Filter which contexts are seen by the LLM for the next question
    contexts = []
    for doc, to_include in zip(qa_session.results, included):
        if to_include:
            contexts.append(doc.page_content)

    qa_session.set_contexts(contexts)
