import uuid
import sqlite3
import hashlib
import hmac
import secrets
from pathlib import Path
from datetime import datetime

import streamlit as st

from scripts.search_hybrid import load_search_resources
from kera.rag import answer_question


# ---------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Kera AI",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --kera-ink: #202124;
        --kera-muted: #777b84;
        --kera-border: #e6e7eb;
        --kera-sidebar: #f7f7f8;
        --kera-accent: #142b4a;
        --kera-accent-soft: #eaf0f7;
    }
    #MainMenu, footer { visibility: hidden; }

/* Mantém o cabeçalho do Streamlit ativo para que o controle de abrir a barra lateral apareça. */
header[data-testid="stHeader"] {
    visibility: visible !important;
    display: block !important;
    background: transparent !important;
}

/* Ícone para reabrir a barra lateral quando ela estiver recolhida. */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
button[kind="header"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    color: #142b4a !important;
}
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
button[kind="header"] {
    background: #142b4a !important;
    border: 1px solid #142b4a !important;
    border-radius: 10px !important;
}
[data-testid="collapsedControl"]:hover,
[data-testid="stSidebarCollapsedControl"]:hover,
button[kind="header"]:hover {
    background: #203d62 !important;
    border-color: #203d62 !important;
}
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg,
button[kind="header"] svg {
    color: #ffffff !important;
    fill: #ffffff !important;
    stroke: #ffffff !important;
}
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: #ffffff !important;
        color: var(--kera-ink) !important;
    }
    /* Garante contraste mesmo quando o tema global do Streamlit é escuro. */
    .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3,
    .stApp [data-testid="stMarkdownContainer"] {
        color: var(--kera-ink);
    }
    section[data-testid="stSidebar"] {
        background: var(--kera-sidebar);
        border-right: 1px solid var(--kera-border);
    }
    section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
    .kera-brand {
        display:flex; align-items:center; gap:11px;
        padding: 5px 8px 16px 8px;
    }
    .kera-mark {
        width:40px; height:40px; border-radius:12px;
        display:flex; align-items:center; justify-content:center;
        background:var(--kera-accent); color:#fff; font-size:23px;
        box-shadow:0 4px 12px rgba(20,43,74,.18);
    }
    .kera-name { font-size:21px; line-height:1.1; font-weight:700; color:#202124; }
    .kera-subtitle { color:#858892; font-size:11px; margin-top:4px; }
    .side-label {
        margin: 14px 8px 8px; color:#858892; font-size:11px;
        font-weight:700; text-transform:uppercase; letter-spacing:.09em;
    }
    .main-wrap { max-width: 900px; margin: 0 auto; }
    .topline {
        position: sticky;
        top: 0;
        z-index: 999;
        background: #ffffff !important;
        padding: 14px 0 18px;
        border-bottom: 1px solid #eceef1;
        margin-bottom: 22px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .topline-title { font-size:19px; font-weight:650; color:#25262a; }
    .topline-caption { font-size:12px; color:#888b93; margin-top:3px; }
    .welcome {
        max-width:720px; margin: 11vh auto 0; text-align:center; padding: 0 20px;
    }
    .welcome-icon {
        margin:0 auto 18px; width:64px; height:64px; border-radius:20px;
        background:var(--kera-accent-soft); color:var(--kera-accent);
        display:flex; align-items:center; justify-content:center; font-size:32px;
    }
    .welcome h1 { font-size:30px; font-weight:650; letter-spacing:-.7px; margin-bottom:10px; }
    .welcome p { font-size:15px; color:#777b84; line-height:1.65; }
    div[data-testid="stChatMessage"] {
        border: 1px solid #eceef1; border-radius:15px; padding:17px 19px;
        margin-bottom:14px; background:#ffffff !important; color:#202124 !important;
    }
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {
        color:#202124 !important;
        opacity:1 !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background:#f8f8f9; border-color:#eeeeef;
    }
    /* Campo de pergunta: fundo claro, texto escuro e placeholder legível. */
    [data-testid="stBottom"], [data-testid="stBottom"] > div {
        background:#ffffff !important;
    }
    div[data-testid="stChatInput"] {
        background:#ffffff !important;
        border:1px solid #dedfe3 !important;
        border-radius:16px !important;
        box-shadow:0 5px 22px rgba(20,24,35,.045);
    }
    div[data-testid="stChatInput"] textarea,
    div[data-testid="stChatInput"] textarea:focus {
        color:#202124 !important;
        -webkit-text-fill-color:#ffffff !important;
        caret-color:#202124 !important;
        background:#202124 !important;
        font-size:14px;
        opacity:1 !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color:#777b84 !important;
        -webkit-text-fill-color:#ffffff !important;
        opacity:1 !important;
    }
    div[data-testid="stChatInput"] button {
        background:#142b4a !important;
        color:#ffffff !important;
        border:0 !important;
    }
    div[data-testid="stChatInput"] button svg { fill:#ffffff !important; }
    /* Botões da barra lateral: inclui especificamente os botões do histórico de chats. */
    section[data-testid="stSidebar"] [data-testid="stButton"] button,
    section[data-testid="stSidebar"] [data-testid="stButton"] button *,
    section[data-testid="stSidebar"] button[kind],
    section[data-testid="stSidebar"] button[kind] * {
        color:#f8fafc !important;
        -webkit-text-fill-color:#f8fafc !important;
        opacity:1 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] button,
    section[data-testid="stSidebar"] button[kind] {
        background-color:#142b4a !important;
        border:1px solid #142b4a !important;
        box-shadow:none !important;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] button:hover,
    section[data-testid="stSidebar"] button[kind]:hover {
        background-color:#203d62 !important;
        border-color:#203d62 !important;
    }
    /* Reforço para texto do botão, mesmo quando o Streamlit adiciona wrappers internos. */
    section[data-testid="stSidebar"] [data-testid="stButton"] button p,
    section[data-testid="stSidebar"] [data-testid="stButton"] button span,
    section[data-testid="stSidebar"] [data-testid="stButton"] button div,
    section[data-testid="stSidebar"] [data-testid="stButton"] button small {
        color:#f8fafc !important;
        -webkit-text-fill-color:#f8fafc !important;
        opacity:1 !important;
    }
    .sidebar-note {
        color:#92959c; font-size:11px; line-height:1.5;
        border-top:1px solid #e7e8eb; padding:13px 8px 0; margin-top:18px;
    }
    .status-dot {
        display:inline-block; width:7px; height:7px; border-radius:50%;
        background:#39a56a; margin-right:6px;
    }
    @media (max-width: 700px) {
        .welcome { margin-top: 6vh; }
        .welcome h1 { font-size:25px; }
    }

/* Oculta apenas o botão Deploy do Streamlit. */
[data-testid="stAppDeployButton"],
.stDeployButton {
    display: none !important;
    visibility: hidden !important;
}


/* Botão para abrir/recolher a barra lateral: azul-marinho em todos os estados. */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
header[data-testid="stHeader"] button[kind="header"] {
    background-color: #142b4a !important;
    background: #142b4a !important;
    border: 1px solid #142b4a !important;
    border-radius: 10px !important;
    opacity: 1 !important;
}
[data-testid="collapsedControl"]:hover,
[data-testid="stSidebarCollapsedControl"]:hover,
[data-testid="stSidebarCollapseButton"]:hover,
header[data-testid="stHeader"] button[kind="header"]:hover {
    background-color: #203d62 !important;
    background: #203d62 !important;
    border-color: #203d62 !important;
}
[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stSidebarCollapseButton"] svg,
header[data-testid="stHeader"] button[kind="header"] svg {
    color: #ffffff !important;
    fill: #ffffff !important;
    stroke: #ffffff !important;
}


    .login-wrap { max-width: 460px; margin: 8vh auto 0; padding: 30px; border: 1px solid #e6e7eb; border-radius: 18px; background: #fff; box-shadow: 0 12px 36px rgba(20,43,74,.08); }
    .login-title { color:#142b4a; font-size:28px; font-weight:750; text-align:center; margin-bottom:6px; }
    .login-subtitle { color:#777b84; text-align:center; margin-bottom:24px; font-size:14px; }
    
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Autenticação e histórico persistente (SQLite)
# ---------------------------------------------------------------------
DB_PATH = Path(__file__).resolve().parent / "kera_users.db"


def _db():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _init_db():
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)


def _password_hash(password: str, salt_hex: str | None = None):
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
    return digest.hex(), salt.hex()


def _verify_password(password: str, salt_hex: str, expected_hash: str) -> bool:
    actual_hash, _ = _password_hash(password, salt_hex)
    return hmac.compare_digest(actual_hash, expected_hash)


def _user_count() -> int:
    with _db() as conn:
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def _create_user(username: str, display_name: str, password: str, role: str = "user"):
    username = username.strip()
    display_name = display_name.strip() or username
    if len(username) < 3:
        raise ValueError("O usuário precisa ter pelo menos 3 caracteres.")
    if len(password) < 10:
        raise ValueError("A senha precisa ter pelo menos 10 caracteres.")
    password_hash, salt = _password_hash(password)
    with _db() as conn:
        conn.execute(
            """INSERT INTO users
               (username, display_name, password_hash, password_salt, role, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (username, display_name, password_hash, salt, role, datetime.now().isoformat(timespec="seconds")),
        )


def _authenticate(username: str, password: str):
    with _db() as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE",
            (username.strip(),),
        ).fetchone()
    if user and _verify_password(password, user["password_salt"], user["password_hash"]):
        return dict(user)
    return None


def _new_conversation(user_id: int):
    now = datetime.now().isoformat(timespec="seconds")
    conversation = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "title": "Nova conversa",
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    with _db() as conn:
        conn.execute(
            "INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (conversation["id"], user_id, conversation["title"], now, now),
        )
    return conversation


def _load_conversations(user_id: int):
    with _db() as conn:
        rows = conn.execute(
            """SELECT id, user_id, title, created_at, updated_at
               FROM conversations WHERE user_id = ?
               ORDER BY updated_at DESC""",
            (user_id,),
        ).fetchall()
        conversations = []
        for row in rows:
            message_rows = conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
                (row["id"],),
            ).fetchall()
            conversations.append({
                "id": row["id"],
                "user_id": row["user_id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "messages": [{"role": m["role"], "content": m["content"]} for m in message_rows],
            })
    return conversations


def _save_message(conversation, role: str, content: str):
    now = datetime.now().isoformat(timespec="seconds")
    with _db() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation["id"], role, content, now),
        )
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (conversation["title"], now, conversation["id"], conversation["user_id"]),
        )
    conversation["updated_at"] = now


def _save_title(conversation):
    now = datetime.now().isoformat(timespec="seconds")
    with _db() as conn:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (conversation["title"], now, conversation["id"], conversation["user_id"]),
        )
    conversation["updated_at"] = now


def _start_new_chat():
    conversation = _new_conversation(st.session_state.kera_user["id"])
    st.session_state.kera_conversations.insert(0, conversation)
    st.session_state.kera_active_chat = conversation["id"]


def _active_conversation():
    active_id = st.session_state.get("kera_active_chat")
    for conversation in st.session_state.kera_conversations:
        if conversation["id"] == active_id:
            return conversation
    if not st.session_state.kera_conversations:
        conversation = _new_conversation(st.session_state.kera_user["id"])
        st.session_state.kera_conversations = [conversation]
        st.session_state.kera_active_chat = conversation["id"]
        return conversation
    conversation = st.session_state.kera_conversations[0]
    st.session_state.kera_active_chat = conversation["id"]
    return conversation


def _logout():
    for key in ("kera_user", "kera_conversations", "kera_active_chat"):
        st.session_state.pop(key, None)
    st.rerun()


_init_db()

# Primeiro acesso: criar o administrador inicial sem senha padrão embutida.
if _user_count() == 0:
    st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">Configurar a Kera AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">Crie a conta de administrador para começar.</div>', unsafe_allow_html=True)
    with st.form("kera_first_admin"):
        admin_name = st.text_input("Nome do administrador")
        admin_username = st.text_input("Usuário de acesso")
        admin_password = st.text_input("Senha (mínimo de 10 caracteres)", type="password")
        admin_password_confirm = st.text_input("Confirme a senha", type="password")
        submitted = st.form_submit_button("Criar administrador", use_container_width=True, type="primary")
    if submitted:
        if admin_password != admin_password_confirm:
            st.error("As senhas não coincidem.")
        else:
            try:
                _create_user(admin_username, admin_name, admin_password, "admin")
                st.success("Administrador criado. Entre com suas credenciais.")
                st.rerun()
            except (ValueError, sqlite3.IntegrityError) as exc:
                st.error(str(exc) if isinstance(exc, ValueError) else "Esse nome de usuário já existe.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# Tela de login
if "kera_user" not in st.session_state:
    st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">Kera AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">Entre para acessar seus chats e o histórico salvo.</div>', unsafe_allow_html=True)
    with st.form("kera_login"):
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        submitted = st.form_submit_button("Entrar", use_container_width=True, type="primary")
    if submitted:
        user = _authenticate(username, password)
        if user:
            st.session_state.kera_user = user
            st.session_state.kera_conversations = _load_conversations(user["id"])
            if not st.session_state.kera_conversations:
                first = _new_conversation(user["id"])
                st.session_state.kera_conversations = [first]
            st.session_state.kera_active_chat = st.session_state.kera_conversations[0]["id"]
            st.rerun()
        else:
            st.error("Usuário ou senha incorretos.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# Garantir que os chats carregados pertencem ao usuário autenticado.
if "kera_conversations" not in st.session_state:
    st.session_state.kera_conversations = _load_conversations(st.session_state.kera_user["id"])
if not st.session_state.kera_conversations:
    first = _new_conversation(st.session_state.kera_user["id"])
    st.session_state.kera_conversations = [first]
if "kera_active_chat" not in st.session_state:
    st.session_state.kera_active_chat = st.session_state.kera_conversations[0]["id"]


# ---------------------------------------------------------------------
# Barra lateral e gestão básica de usuários
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="kera-brand">
            <div class="kera-mark">✳</div>
            <div>
                <div class="kera-name">Kera AI</div>
                <div class="kera-subtitle">Inteligência para gestão de clínicas</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Conectado: {st.session_state.kera_user['display_name']}")

    if st.button("＋  Nova conversa", use_container_width=True, type="primary"):
        _start_new_chat()
        st.rerun()

    st.markdown('<div class="side-label">Histórico de conversas</div>', unsafe_allow_html=True)
    for conversation in st.session_state.kera_conversations:
        label = conversation["title"]
        if len(label) > 31:
            label = label[:28].rstrip() + "…"
        is_active = conversation["id"] == st.session_state.kera_active_chat
        button_label = ("●  " if is_active else "   ") + label
        if st.button(
            button_label,
            key=f"chat_{conversation['id']}",
            use_container_width=True,
            help=conversation["title"],
            type="secondary" if is_active else "tertiary",
        ):
            st.session_state.kera_active_chat = conversation["id"]
            st.rerun()

    if st.session_state.kera_user["role"] == "admin":
        with st.expander("Administrar usuários"):
            with st.form("kera_create_user"):
                new_display_name = st.text_input("Nome")
                new_username = st.text_input("Novo usuário")
                new_password = st.text_input("Senha inicial (mín. 10 caracteres)", type="password")
                create_user = st.form_submit_button("Criar usuário", use_container_width=True)
            if create_user:
                try:
                    _create_user(new_username, new_display_name, new_password, "user")
                    st.success("Usuário criado.")
                    st.rerun()
                except (ValueError, sqlite3.IntegrityError) as exc:
                    st.error(str(exc) if isinstance(exc, ValueError) else "Esse nome de usuário já existe.")

            with _db() as conn:
                users = conn.execute("SELECT username, display_name, role, created_at FROM users ORDER BY id").fetchall()
            st.caption("Contas cadastradas")
            for item in users:
                st.write(f"**{item['display_name']}** — `{item['username']}` ({item['role']})")

    if st.button("Sair da conta", use_container_width=True):
        _logout()

    st.markdown(
        """
        <div class="sidebar-note">
            <span class="status-dot"></span>Assistente pronto para uso<br>
            Histórico salvo na base local desta instalação.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Carregamento da busca
# ---------------------------------------------------------------------
@st.cache_resource
def get_search_resources():
    return load_search_resources()


try:
    resources = get_search_resources()
except Exception as exc:
    st.error(
        "Não foi possível carregar os recursos de busca. "
        "Verifique a coleção Chroma e os modelos de retrieval."
    )
    with st.expander("Detalhes técnicos"):
        st.exception(exc)
    st.stop()


conversation = _active_conversation()

# Se a primeira pergunta foi enviada, usa-a como título persistente do histórico.
def _update_title(conv, question):
    if conv["title"] == "Nova conversa":
        clean = " ".join(question.split())
        conv["title"] = clean[:42] + ("…" if len(clean) > 42 else "")
        _save_title(conv)


# ---------------------------------------------------------------------
# Área principal
# ---------------------------------------------------------------------
st.markdown('<div class="main-wrap">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="topline">
        <div>
            <div class="topline-title">{conversation["title"]}</div>
            <div class="topline-caption">Assistente de gestão operacional e comercial</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not conversation["messages"]:
    st.markdown(
        """
        <div class="welcome">
            <div class="welcome-icon">✳</div>
            <h1>Como posso ajudar hoje?</h1>
            <p>Faça uma pergunta sobre gestão operacional, indicadores comerciais,
            atendimento ou processos da clínica. As respostas usam os documentos
            indexados na base da Kera AI.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in conversation["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Pergunte algo sobre a operação da clínica…")

if question:
    _update_title(conversation, question)
    conversation["messages"].append({"role": "user", "content": question})
    _save_message(conversation, "user", question)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analisando os documentos…"):
            try:
                result = answer_question(question, resources)
                answer = result["answer"]
                conversation["messages"].append({
                    "role": "assistant",
                    "content": answer,
                })
                _save_message(conversation, "assistant", answer)
                st.markdown(answer)
            except Exception as exc:
                st.error(
                    "Não foi possível gerar a resposta. Verifique o Ollama, "
                    "o modelo configurado e os recursos de busca."
                )
                with st.expander("Detalhes técnicos"):
                    st.exception(exc)

    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
