from typing import List
import streamlit as st

from llm.qa import model_configs
from llm.qa.document_store import DocStore
from llm.qa.embedding import QAEmbeddings
from llm.qa.fastchatter import MultithreadChat, QASession

st.set_page_config(page_title="pubmed chat", page_icon=":robot_face:", layout='wide')


@st.cache_resource
def get_chatter():
    # Chat model shared by all streamlit sessions
    return MultithreadChat(model_configs.VICUNA)


@st.cache_resource
def get_docstore():
    # DocStore shared by all streamlit sessions
    # TODO: index handling
    return DocStore(QAEmbeddings(), index_name="KubernetesConcepts")


def get_qa_session(chatter: MultithreadChat, docstore: DocStore) -> QASession:
    # Conversation/contexts for each streamlit session
    if "qa_session" not in st.session_state:
        qa_session = QASession(chatter, docstore)
        st.session_state["qa_session"] = qa_session
        return qa_session
    return st.session_state["qa_session"]


def apply_filter():
    contexts = []
    for doc, to_include in zip(qa_session.results, included):
        if to_include:
            contexts.append(doc["text"])

    qa_session.clear(keep_results=True)
    qa_session.set_context(contexts)
    qa_session.append_question(user_input)


chatter = get_chatter()
docstore = get_docstore()
qa_session = get_qa_session(chatter, docstore)
output = st.text('')
included: List[bool] = []

clear_convo = st.button("clear conversation", on_click=qa_session.clear)

with st.form(key='my_form', clear_on_submit=True):
    user_input = st.text_area("You:", key='input', height=100)
    question_submit_button = st.form_submit_button(label='Send')

if question_submit_button and not clear_convo:
    qa_session.append_question(user_input)
    output.write(qa_session.get_history())
    with st.spinner("Searching..."):
        qa_session.update_context(user_input)

if qa_session.results:
    with st.form(key='checklists'):
        included = []
        for idx, doc in enumerate(qa_session.results):
            include = st.checkbox('include in chat context', key=idx, value=True)
            included.append(include)
            st.write(doc["text"])
            st.write({k: v for k, v in doc.items() if k != "text"})
            st.divider()
        checklist_submit_button = st.form_submit_button(label='Filter')

    if checklist_submit_button:
        apply_filter()
else:
    checklist_submit_button = False

if (question_submit_button or checklist_submit_button) and not clear_convo:
    message_string = qa_session.get_history()
    for text in qa_session.conversation_stream():
        message = message_string + text
        message = message.replace('\n', '\n\n')
        output.write(message)
