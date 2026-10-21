import sqlite3
import streamlit as st

from kera.database import create_user, list_users


def render_admin():
    user = st.session_state.get("kera_user")
    if not user:
        st.switch_page("pages/login.py")
    if user.get("role") != "admin":
        st.error("Você não tem permissão para acessar esta página.")
        if st.button("Voltar para Home"):
            st.switch_page("pages/home.py")
        st.stop()

    st.title("Administrar usuários")
    st.caption("Somente administradores podem cadastrar contas de acesso.")
    if st.button("← Voltar para Home"):
        st.switch_page("pages/home.py")

    with st.form("kera_create_user"):
        name = st.text_input("Nome")
        username = st.text_input("Novo usuário")
        password = st.text_input("Senha inicial (mínimo de 10 caracteres)", type="password")
        submitted = st.form_submit_button("Criar usuário", type="primary")
    if submitted:
        try:
            create_user(username, name, password, "user")
            st.success("Usuário criado.")
            st.rerun()
        except (ValueError, sqlite3.IntegrityError) as exc:
            st.error(str(exc) if isinstance(exc, ValueError) else "Esse nome de usuário já existe.")

    st.subheader("Contas cadastradas")
    for item in list_users():
        st.write(f"**{item['display_name']}** — `{item['username']}` ({item['role']})")
