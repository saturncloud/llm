import streamlit as st
from llm.prompt import Conversation, Message, Prompt

from examples.streamlit_ui.components import (
    chat_bubble,
    generation_settings,
    get_engine,
    setup_page,
)


def get_conversation() -> Conversation:
    if "chat_conversation" in st.session_state:
        return st.session_state["chat_conversation"]

    conversation = Conversation()
    st.session_state["chat_conversation"] = conversation
    return conversation


def get_prompt() -> Prompt:
    if "chat_prompt" in st.session_state:
        return st.session_state["chat_prompt"]
    prompt = Prompt.from_model_config(
        st.session_state.model_config,
        system_message="You are a helpful assistant having a conversation with a curious user. Be courteous, and help the user with their requests. Do not make up information if you do not know.",
    )
    st.session_state["chat_prompt"] = prompt
    return prompt


# Ensure only the main proc interacts with streamlit
if __name__ == "__main__":
    setup_page("Chat")
    engine = get_engine()
    conversation = get_conversation()
    prompt = get_prompt()

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
            system_message = st.text_area("System Message", value=prompt.system_message)
            generation_kwargs = generation_settings()
            apply_settings = st.form_submit_button(label="Apply")
            if apply_settings:
                prompt.system_message = system_message

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
            prompt_str = conversation.render(prompt)
            for text in engine.generate_stream(
                prompt_str, stop=prompt.stop_strings, **generation_kwargs
            ):
                answer.text(text)
                input_message.response = text
