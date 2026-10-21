import streamlit as st
from kera.chat import render_chat

if not st.session_state.get("kera_user"):
    st.switch_page("pages/login.py")
render_chat()
