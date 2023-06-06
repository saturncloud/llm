import time
import uuid

import streamlit as st

output = st.empty()

with st.form(key='my_form', clear_on_submit=True):
    user_input = st.text_area("You:", key='input', height=100)
    submit_button = st.form_submit_button(label='Send')

output.write(user_input)