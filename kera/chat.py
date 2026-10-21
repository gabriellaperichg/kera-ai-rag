import streamlit as st

from scripts.search_hybrid import load_search_resources
from kera.rag import answer_question
from kera.auth import logout
from kera.database import (
    create_user, db, list_users, load_conversations, new_conversation,
    save_message, save_title
)


def format_answer_for_display(answer):
    """
    Normaliza respostas que chegam como tabela Markdown dentro de bloco de código.
    Mantém blocos de código reais, mas remove cercas em torno de tabelas Markdown.
    """
    if not isinstance(answer, str):
        answer = str(answer)

    text = answer.strip()
    # Modelos às vezes cercam uma tabela Markdown com ```markdown ou ```text.
    fence_pattern = re.compile(
        r"```(?:markdown|md|text)?\s*\n(.*?)\n```",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def unwrap_table(match):
        block = match.group(1).strip()
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        has_header = any("|" in line for line in lines[:3])
        has_separator = any(
            "|" in line and re.search(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", line)
            for line in lines[:5]
        )
        if has_header and has_separator:
            return block
        return match.group(0)

    text = fence_pattern.sub(unwrap_table, text)

    # Remove cercas simples quando o conteúdo inteiro é claramente uma tabela Markdown.
    if text.startswith("```") and text.endswith("```"):
        inner = re.sub(r"^```[^\n]*\n?", "", text)
        inner = re.sub(r"\n?```$", "", inner).strip()
        lines = [line.strip() for line in inner.splitlines() if line.strip()]
        if (
            len(lines) >= 2
            and "|" in lines[0]
            and any("|" in line and re.search(r"-{3,}", line) for line in lines[1:3])
        ):
            text = inner

    # Se a tabela foi produzida com pipes e separadores inconsistentes,
    # normaliza espaços, sem alterar os valores ou inventar colunas.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            # Mantém o Markdown como veio; o Streamlit renderiza a tabela nativamente.
            lines[i] = stripped
    return "\n".join(lines)


def render_assistant_answer(answer):
    formatted = format_answer_for_display(answer)
    st.markdown(formatted, unsafe_allow_html=False)


@st.cache_resource
def get_search_resources():
    return load_search_resources()


def ensure_user_state():
    user = st.session_state.get("kera_user")
    if not user:
        st.switch_page("pages/login.py")
    if "kera_conversations" not in st.session_state:
        st.session_state.kera_conversations = load_conversations(user["id"])
    if not st.session_state.kera_conversations:
        st.session_state.kera_conversations = [new_conversation(user["id"])]
    if "kera_active_chat" not in st.session_state:
        st.session_state.kera_active_chat = st.session_state.kera_conversations[0]["id"]
    # Defesa adicional contra estado antigo/inconsistente: somente conversas deste usuário.
    st.session_state.kera_conversations = [
        c for c in st.session_state.kera_conversations if c.get("user_id") == user["id"]
    ]


def active_conversation():
    conversations = st.session_state.kera_conversations
    active_id = st.session_state.get("kera_active_chat")
    for conv in conversations:
        if conv["id"] == active_id:
            return conv
    if not conversations:
        conv = new_conversation(st.session_state.kera_user["id"])
        st.session_state.kera_conversations = [conv]
        st.session_state.kera_active_chat = conv["id"]
        return conv
    st.session_state.kera_active_chat = conversations[0]["id"]
    return conversations[0]


def start_new_chat():
    conv = new_conversation(st.session_state.kera_user["id"])
    st.session_state.kera_conversations.insert(0, conv)
    st.session_state.kera_active_chat = conv["id"]


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """<div class="kera-brand"><div class="kera-mark">✳</div><div>
            <div class="kera-name">Kera AI</div>
            <div class="kera-subtitle">Inteligência para gestão de clínicas</div>
            </div></div>""",
            unsafe_allow_html=True,
        )
        st.caption(f"Conectado: {st.session_state.kera_user['display_name']}")
        if st.button("＋  Nova conversa", use_container_width=True, type="primary"):
            start_new_chat()
            st.rerun()

        st.markdown('<div class="side-label">Histórico de conversas</div>', unsafe_allow_html=True)
        for conv in st.session_state.kera_conversations:
            label = conv["title"]
            if len(label) > 31:
                label = label[:28].rstrip() + "…"
            active = conv["id"] == st.session_state.kera_active_chat
            if st.button(("●  " if active else "   ") + label,
                         key=f"chat_{conv['id']}", use_container_width=True,
                         help=conv["title"], type="secondary" if active else "tertiary"):
                st.session_state.kera_active_chat = conv["id"]
                st.rerun()

        if st.session_state.kera_user["role"] == "admin":
            if st.button("⚙  Administrar usuários", use_container_width=True):
                st.switch_page("pages/admin.py")

        if st.button("Sair da conta", use_container_width=True):
            logout()

        st.markdown(
            '<div class="sidebar-note"><span class="status-dot"></span>Assistente pronto para uso<br>'
            'Histórico salvo na base local desta instalação.</div>',
            unsafe_allow_html=True,
        )


def render_chat():
    ensure_user_state()
    render_sidebar()
    try:
        resources = get_search_resources()
    except Exception as exc:
        st.error("Não foi possível carregar os recursos de busca. Verifique a coleção Chroma e os modelos de retrieval.")
        with st.expander("Detalhes técnicos"):
            st.exception(exc)
        st.stop()

    conv = active_conversation()
    st.markdown('<div class="main-wrap">', unsafe_allow_html=True)
    st.markdown(
        f"""<div class="topline"><div><div class="topline-title">{conv["title"]}</div>
        <div class="topline-caption">Assistente de gestão operacional e comercial</div></div></div>""",
        unsafe_allow_html=True,
    )
    if not conv["messages"]:
        st.markdown(
            """<div class="welcome"><div class="welcome-icon">✳</div>
            <h1>Como posso ajudar hoje?</h1>
            <p>Faça uma pergunta sobre gestão operacional, indicadores comerciais,
            atendimento ou processos da clínica. As respostas usam os documentos
            indexados na base da Kera AI.</p></div>""",
            unsafe_allow_html=True,
        )

    for message in conv["messages"]:
        with st.chat_message(message["role"]):
            render_assistant_answer(message["content"])

    question = st.chat_input("Pergunte algo sobre a operação da clínica…")
    if question:
        if conv["title"] == "Nova conversa":
            clean = " ".join(question.split())
            conv["title"] = clean[:42] + ("…" if len(clean) > 42 else "")
            save_title(conv)
        conv["messages"].append({"role": "user", "content": question})
        save_message(conv, "user", question)
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Analisando os documentos…"):
                try:
                    result = answer_question(question, resources)
                    answer = result["answer"]
                    conv["messages"].append({"role": "assistant", "content": answer})
                    save_message(conv, "assistant", answer)
                    render_assistant_answer(answer)
                except Exception as exc:
                    st.error("Não foi possível gerar a resposta. Verifique o Ollama, o modelo configurado e os recursos de busca.")
                    with st.expander("Detalhes técnicos"):
                        st.exception(exc)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
