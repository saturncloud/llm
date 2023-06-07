from typing import List
import streamlit as st

from bert_qa.fastchatter import StreamlitChatLoop

st.set_page_config(page_title="pubmed chat", page_icon=":robot_face:", layout='wide')

@st.cache_resource
def get_chatter():
    return StreamlitChatLoop()

chatter = get_chatter()

print('GOO')

output = st.text('')
included: List[bool] = []

def clear_conversation():
    global included, user_input
    chatter.conv.system = ""
    chatter.conv.messages = []
    chatter.results = []
    included = []


def apply_filter():
    print('FILTER')
    docs = []
    for doc, to_include in zip(chatter.results, included):
        if to_include:
            docs.append(doc)
    chatter.make_prompt(docs)
    chatter.conv.messages = []
    chatter.append_question(user_input)

clear_convo = st.button("clear conversation", on_click=clear_conversation)

with st.form(key='my_form', clear_on_submit=True):
    user_input = st.text_area("You:", key='input', height=100)
    question_submit_button = st.form_submit_button(label='Send')

if question_submit_button and not clear_convo:
    chatter.append_question(user_input)
    output.write(chatter.get_chat())
    with st.spinner("Searching..."):
        chatter.search_results(user_input)

if chatter.results:
    with st.form(key='checklists'):
        included = []
        for idx, doc in enumerate(chatter.results):
            include = st.checkbox('include in chat context', key=idx, value=True)
            included.append(include)
            st.write(f"Title - {doc.title}")
            st.write(f"Publication - {doc.meta['publication_title']}")
            # text = doc.text.replace('\n', '')
            text = doc.original_text
            st.write(f"Text - {text}")
            st.divider()
        checklist_submit_button = st.form_submit_button(label='Filter')

    if checklist_submit_button:
        apply_filter()
else:
    checklist_submit_button = False

if (question_submit_button or checklist_submit_button) and not clear_convo:
    message_string = chatter.get_chat()
    for text in chatter.loop():
        message = message_string + text
        message = message.replace('\n', '\n\n')
        output.write(message)
