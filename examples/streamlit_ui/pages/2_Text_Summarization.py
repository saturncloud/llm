import streamlit as st

from llm.summarization.prompts import TextSummarization
from llm.prompt import Conversation, Message

from examples.streamlit_ui.components import chat_bubble, generation_settings, get_engine, setup_page

def get_conversation() -> Conversation:
    if "summary_conversation" in st.session_state:
        return st.session_state["summary_conversation"]

    conversation = Conversation()
    st.session_state["summary_conversation"] = conversation
    return conversation


if __name__ == "__main__":
    setup_page("Text Summarization")

    engine = get_engine()
    conversation = get_conversation()
    prompt = TextSummarization.from_model_config(st.session_state.model_config)

    chat_container = st.container()
    clear_convo = st.button("Clear")
    if clear_convo:
        conversation.clear()

    with st.form(key="input_form", clear_on_submit=True):
        user_input = st.text_area("Text:", key="input", height=100)
        query_submitted = st.form_submit_button(label="Submit")
        if clear_convo or (query_submitted and not user_input):
            query_submitted = False
        input_message = Message(user_input)

    with st.sidebar:
        st.header("Settings")
        with st.form(key="settings_form", clear_on_submit=False):
            generation_kwargs = generation_settings()
            st.form_submit_button(label="Apply")

    # Write current chat
    if query_submitted:
        conversation.add(input_message)

    with chat_container:
        for message in conversation.messages:
            chat_bubble("user", message.input)
            if message.response is not None:
                chat_bubble("assistant", message.response)

    if query_submitted:
        # Stream response from LLM, updating chat window at each step
        with chat_container:
            answer = chat_bubble("assistant")
            # No history in text summary, render only the latest input message
            prompt_str = prompt.render([input_message])
            for text in engine.generate_stream(prompt_str, stop=prompt.stop_strings, **generation_kwargs):
                answer.write(text)
                input_message.response = text
