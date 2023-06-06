import streamlit as st

from bert_qa.fastchatter import StreamlitChatLoop

st.set_page_config(page_title="pubmed chat", page_icon=":robot_face:", layout='wide')

@st.cache_resource
def get_chatter():
    chatter = StreamlitChatLoop()
    chatter.load_models()
    return chatter


# if 'chatter' not in st.session_state:
#     st.session_state['chatter'] = StreamlitChatLoop()
#     st.session_state['chatter'].load_models()
#
# chatter = st.session_state['chatter']

chatter = get_chatter()

print('GOO')

output = st.text('')

def clear_conversation():
    chatter.conv = None
    st.session_state.pop('user_input')
    chatter.question = None
    chatter.results = None
    chatter.system = None

st.button("clear conversation", on_click=clear_conversation)


with st.form(key='my_form', clear_on_submit=True):
    user_input = st.text_area("You:", key='input', height=100)
    submit_button = st.form_submit_button(label='Send')
    st.session_state['user_input'] = user_input

if not st.session_state.get('user_input') or st.session_state.get('last_input') == st.session_state.get('user_input'):
    st.stop()

if chatter.conv is None:
    chatter.set_question(st.session_state['user_input'])
    chatter.make_prompt(chatter.results)
    chatter.create_conversation()


include_doc = []

def apply_filter():
    print('FILTER')
    docs = []
    for doc, to_include in zip(chatter.results, include_doc):
        if to_include:
            docs.append(doc)
    chatter.make_prompt(docs)
    chatter.conv.messages = []
    chatter.conv.system = chatter.system
    st.session_state['user_input'] = chatter.question


with st.form(key='checklists'):
    for idx, doc in enumerate(chatter.results):
        include = st.checkbox('include in chat context', key=idx, value=True)
        include_doc.append(include)
        st.write(f"Title - {doc.title}")
        st.write(f"Publication - {doc.meta['publication_title']}")
        # text = doc.text.replace('\n', '')
        text = doc.original_text
        # print(idx, text)
        st.write(f"Text - {text}")
        st.divider()
    print(include_doc)
    submit_button = st.form_submit_button(label='Filter', on_click=apply_filter)


chatter.start_loop(st.session_state['user_input'])
messages = [f'{role}: {"" if message is None else message}' for role, message in chatter.conv.messages]
message_string = "\n\n".join(messages)
for text in chatter.loop():
    message = message_string + text
    message = message.replace('\n', '\n\n')
    output.write(message)

st.session_state['last_input'] = st.session_state['user_input']


