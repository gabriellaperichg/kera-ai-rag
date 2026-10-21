import streamlit as st

from kera.database import init_db
from kera.styles import load_styles

st.set_page_config(
    page_title="Kera AI",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_styles()
init_db()

if not st.session_state.get("kera_user"):
    page = st.Page("pages/login.py", title="Login", icon=":material/login:", default=True)
    nav = st.navigation([page], position="hidden")
else:
    pages = [
        st.Page("pages/home.py", title="Home", icon=":material/chat:", default=True),
    ]
    if st.session_state.kera_user.get("role") == "admin":
        pages.append(st.Page("pages/admin.py", title="Administrar usuários", icon=":material/group:"))
    nav = st.navigation(pages, position="hidden")
nav.run()
