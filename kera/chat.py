import re
import threading
import time
import streamlit as st

from scripts.search_hybrid import load_search_resources
from kera.rag import answer_question
from kera.llm import GenerationCancelled
from kera.auth import logout
from kera.database import (
    load_conversations, new_conversation, save_message, save_title,
    rename_conversation, delete_conversation
)


def format_answer_for_display(answer):
    """Retira cercas de código em volta de tabelas Markdown, preservando outros códigos."""
    if not isinstance(answer, str):
        answer = str(answer)
    text = answer.strip()
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
        return block if has_header and has_separator else match.group(0)

    text = fence_pattern.sub(unwrap_table, text)
    if text.startswith("```") and text.endswith("```"):
        inner = re.sub(r"^```[^\n]*\n?", "", text)
        inner = re.sub(r"\n?```$", "", inner).strip()
        lines = [line.strip() for line in inner.splitlines() if line.strip()]
        if len(lines) >= 2 and "|" in lines[0] and any("|" in line and re.search(r"-{3,}", line) for line in lines[1:3]):
            text = inner
    return text


def render_assistant_answer(answer):
    st.markdown(format_answer_for_display(answer), unsafe_allow_html=False)


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
        if st.button("＋  Nova conversa", use_container_width=True, type="primary",
                     disabled=bool(st.session_state.get("kera_job", {}).get("status") == "running")):
            start_new_chat()
            st.rerun()

        st.markdown('<div class="side-label">Histórico de conversas</div>', unsafe_allow_html=True)
        job_running = bool(
            st.session_state.get("kera_job", {}).get("status") in ("running", "cancelling")
        )
        for conv in list(st.session_state.kera_conversations):
            label = conv["title"]
            if len(label) > 31:
                label = label[:28].rstrip() + "…"
            active = conv["id"] == st.session_state.kera_active_chat
            if st.button(("●  " if active else "   ") + label,
                         key=f"chat_{conv['id']}", use_container_width=True,
                         help=conv["title"], type="secondary" if active else "tertiary",
                         disabled=job_running):
                st.session_state.kera_active_chat = conv["id"]
                st.rerun()

            with st.expander("⋯ Gerenciar conversa", expanded=False):
                with st.form(key=f"rename_form_{conv['id']}"):
                    new_title = st.text_input(
                        "Nome da conversa", value=conv["title"], max_chars=100,
                        key=f"rename_input_{conv['id']}", disabled=job_running,
                    )
                    rename_submitted = st.form_submit_button(
                        "Salvar nome", use_container_width=True, disabled=job_running
                    )
                if rename_submitted:
                    try:
                        if rename_conversation(
                            st.session_state.kera_user["id"], conv["id"], new_title
                        ):
                            conv["title"] = " ".join(new_title.split()).strip()[:100]
                            st.success("Conversa renomeada.")
                            st.rerun()
                        else:
                            st.error("Não foi possível encontrar essa conversa para renomear.")
                    except ValueError as exc:
                        st.error(str(exc))

                pending_id = st.session_state.get("kera_pending_delete_chat")
                if pending_id == conv["id"]:
                    st.warning(f'Excluir “{conv["title"]}” e todas as mensagens dela?')
                    confirm_col, cancel_col = st.columns(2)
                    with confirm_col:
                        if st.button(
                            "Excluir", key=f"confirm_delete_{conv['id']}",
                            type="primary", use_container_width=True, disabled=job_running,
                        ):
                            deleted = delete_conversation(
                                st.session_state.kera_user["id"], conv["id"]
                            )
                            st.session_state.kera_conversations = [
                                item for item in st.session_state.kera_conversations
                                if item["id"] != conv["id"]
                            ]
                            st.session_state.kera_pending_delete_chat = None
                            if st.session_state.get("kera_active_chat") == conv["id"]:
                                st.session_state.kera_active_chat = (
                                    st.session_state.kera_conversations[0]["id"]
                                    if st.session_state.kera_conversations else None
                                )
                            if not st.session_state.kera_conversations:
                                fresh = new_conversation(st.session_state.kera_user["id"])
                                st.session_state.kera_conversations = [fresh]
                                st.session_state.kera_active_chat = fresh["id"]
                            if deleted:
                                st.toast("Conversa excluída.")
                            st.rerun()
                    with cancel_col:
                        if st.button(
                            "Cancelar", key=f"cancel_delete_{conv['id']}",
                            use_container_width=True,
                        ):
                            st.session_state.kera_pending_delete_chat = None
                            st.rerun()
                else:
                    if st.button(
                        "Excluir conversa…", key=f"request_delete_{conv['id']}",
                        use_container_width=True, disabled=job_running,
                    ):
                        st.session_state.kera_pending_delete_chat = conv["id"]
                        st.rerun()

        if st.session_state.kera_user["role"] == "admin":
            if st.button("⚙  Administrar usuários", use_container_width=True,
                         disabled=bool(st.session_state.get("kera_job", {}).get("status") == "running")):
                st.switch_page("pages/admin.py")

        if st.button("Sair da conta", use_container_width=True,
                     disabled=bool(st.session_state.get("kera_job", {}).get("status") == "running")):
            logout()

        st.markdown(
            '<div class="sidebar-note"><span class="status-dot"></span>Assistente pronto para uso<br>'
            'Histórico salvo na base local desta instalação.</div>',
            unsafe_allow_html=True,
        )


def _run_answer_job(job, question, resources):
    """Worker de fundo: nunca chama Streamlit; apenas atualiza o dicionário do trabalho."""
    try:
        result = answer_question(question, resources, cancel_event=job["cancel_event"])
        if job["cancel_event"].is_set():
            job["status"] = "cancelled"
        else:
            job["result"] = result
            job["status"] = "done"
    except GenerationCancelled:
        job["status"] = "cancelled"
    except Exception as exc:
        job["error"] = f"{type(exc).__name__}: {exc}"
        job["status"] = "error"
    finally:
        job["finished_at"] = time.time()


def _start_answer_job(question, conversation, resources):
    if conversation["title"] == "Nova conversa":
        clean = " ".join(question.split())
        conversation["title"] = clean[:42] + ("…" if len(clean) > 42 else "")
        save_title(conversation)
    conversation["messages"].append({"role": "user", "content": question})
    save_message(conversation, "user", question)

    job = {
        "status": "running",
        "question": question,
        "conversation_id": conversation["id"],
        "cancel_event": threading.Event(),
        "result": None,
        "error": None,
        "started_at": time.time(),
        "finished_at": None,
    }
    st.session_state.kera_job = job
    worker = threading.Thread(
        target=_run_answer_job,
        args=(job, question, resources),
        daemon=True,
        name="kera-answer-worker",
    )
    job["thread"] = worker
    worker.start()


def _finish_job_if_needed():
    job = st.session_state.get("kera_job")
    if not job or job.get("status") not in ("done", "cancelled", "error"):
        return
    if job.get("handled"):
        return

    conv = next(
        (c for c in st.session_state.kera_conversations if c["id"] == job["conversation_id"]),
        None,
    )
    if job["status"] == "done" and conv is not None:
        answer = (job.get("result") or {}).get("answer", "")
        if answer:
            conv["messages"].append({"role": "assistant", "content": answer})
            save_message(conv, "assistant", answer)
    job["handled"] = True


@st.fragment(run_every=0.5)
def _render_composer(conversation, resources):
    """
    Composer que se atualiza enquanto a resposta é gerada.
    O botão da própria barra alterna entre enviar (↑) e cancelar (■).
    """
    job = st.session_state.get("kera_job")
    busy = bool(job and job.get("status") in ("running", "cancelling"))

    with st.form("kera_question_form", clear_on_submit=True, border=False):
        col_input, col_action = st.columns([12, 1], vertical_alignment="bottom")
        with col_input:
            question = st.text_input(
                "Pergunta",
                placeholder="Pergunte algo sobre a operação da clínica…",
                label_visibility="collapsed",
                key="kera_question_input",
                disabled=busy,
            )
        with col_action:
            action_label = "■" if busy else "↑"
            action_help = "Cancelar geração" if busy else "Enviar pergunta"
            submitted = st.form_submit_button(
                action_label,
                type="primary",
                help=action_help,
                disabled=(job is not None and job.get("status") == "cancelling"),
                use_container_width=True,
            )

    if submitted and busy:
        job["cancel_event"].set()
        job["status"] = "cancelling"
        st.rerun()
    elif submitted and question.strip() and not busy:
        _start_answer_job(question.strip(), conversation, resources)
        st.rerun()

    job = st.session_state.get("kera_job")
    if job:
        status = job.get("status")
        if status == "running":
            st.markdown(
                '<div class="kera-working"><span class="kera-spinner"></span> Kera AI está pensando…</div>',
                unsafe_allow_html=True,
            )
        elif status == "cancelling":
            st.caption("Cancelando geração…")
        elif status in ("done", "cancelled", "error"):
            # Se a mudança ocorreu durante o rerun do fragmento, pede um rerun
            # completo para persistir o resultado. Depois de tratado, não cria loop.
            if not job.get("handled"):
                st.rerun()
            elif status == "cancelled" and not job.get("notice_shown"):
                st.info("Geração cancelada. Você pode enviar outra pergunta.")
                job["notice_shown"] = True
            elif status == "error" and not job.get("notice_shown"):
                st.error("Não foi possível gerar a resposta. Verifique o Ollama e os recursos de busca.")
                with st.expander("Detalhes técnicos"):
                    st.code(job.get("error") or "Erro sem detalhes.")
                job["notice_shown"] = True

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
    st.markdown(
        f"""<div class="topline"><div><div class="topline-title">{conv["title"]}</div>
        <div class="topline-caption">Assistente de gestão operacional e comercial</div></div></div>""",
        unsafe_allow_html=True,
    )

    job = st.session_state.get("kera_job")
    _finish_job_if_needed()

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
            if message["role"] == "assistant":
                render_assistant_answer(message["content"])
            else:
                st.markdown(message["content"])

    _render_composer(conv, resources)
