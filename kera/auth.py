import sqlite3
import streamlit as st

from kera.database import authenticate, create_user, load_conversations, new_conversation, user_count


def logout():
    for key in ("kera_user", "kera_conversations", "kera_active_chat"):
        st.session_state.pop(key, None)
    st.rerun()


def render_login():
    if user_count() == 0:
        left, center, right = st.columns([1, 1.15, 1])
        with center:
            st.markdown(
                '<div class="login-brand"><div class="login-mark">✳</div>'
                '<div class="login-title">Configurar a Kera AI</div>'
                '<div class="login-subtitle">Crie a conta de administrador para proteger o acesso e gerenciar os usuários.</div></div>',
                unsafe_allow_html=True,
            )
            with st.form("kera_first_admin"):
                admin_name = st.text_input("Nome do administrador")
                admin_username = st.text_input("Usuário de acesso")
                admin_password = st.text_input("Senha (mínimo de 10 caracteres)", type="password")
                confirm = st.text_input("Confirme a senha", type="password")
                submitted = st.form_submit_button("Criar administrador", use_container_width=True, type="primary")
            if submitted:
                if admin_password != confirm:
                    st.error("As senhas não coincidem.")
                else:
                    try:
                        create_user(admin_username, admin_name, admin_password, "admin")
                        st.success("Administrador criado. Entre com suas credenciais.")
                        st.rerun()
                    except (ValueError, sqlite3.IntegrityError) as exc:
                        st.error(str(exc) if isinstance(exc, ValueError) else "Esse nome de usuário já existe.")
        return

    left, center, right = st.columns([1, 1.15, 1])
    with center:
        st.markdown(
            '<div class="login-brand"><div class="login-mark">✳</div>'
            '<div class="login-title">Kera AI</div>'
            '<div class="login-subtitle">Entre na sua conta para acessar suas conversas e continuar de onde parou.</div></div>',
            unsafe_allow_html=True,
        )
        with st.form("kera_login"):
            username = st.text_input("Usuário", placeholder="Digite seu usuário")
            password = st.text_input("Senha", type="password", placeholder="Digite sua senha")
            submitted = st.form_submit_button("Entrar", use_container_width=True, type="primary")
        st.markdown(
            '<div style="text-align:center;color:#8992a0;font-size:12px;margin-top:16px;">'
            'Acesso restrito a usuários autorizados</div>',
            unsafe_allow_html=True,
        )
    if submitted:
        user = authenticate(username, password)
        if user:
            st.session_state.kera_user = user
            st.session_state.kera_conversations = load_conversations(user["id"])
            if not st.session_state.kera_conversations:
                st.session_state.kera_conversations = [new_conversation(user["id"])]
            st.session_state.kera_active_chat = st.session_state.kera_conversations[0]["id"]
            st.rerun()
        else:
            st.error("Usuário ou senha incorretos.")
