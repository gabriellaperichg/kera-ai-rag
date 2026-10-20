
import streamlit as st

from scripts.search_hybrid import load_search_resources
from kera.rag import answer_question


st.set_page_config(
    page_title="Kera AI",
    page_icon="💬",
    layout="wide",
)

st.title("Kera AI")
st.caption("Assistente de gestão operacional e comercial de clínicas")

st.info(
    "As respostas são baseadas nos documentos e nas transcrições "
    "indexados. Confira as fontes antes de tomar decisões."
)


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


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])

            if sources:
                with st.expander("Fontes consultadas"):
                    for source in sources:
                        st.markdown(
                            f"**[{source['id']}] "
                            f"{source['document_name']}**"
                        )
                        st.caption(
                            f"{source['source_label']} · "
                            f"Páginas: {source['pages']}"
                        )
                        st.text(source["text"])
                        st.divider()


question = st.chat_input("Faça uma pergunta sobre a operação da clínica")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Consultando as fontes e gerando a resposta..."):
            try:
                result = answer_question(question, resources)
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:
                    with st.expander("Fontes consultadas"):
                        for source in sources:
                            st.markdown(
                                f"**[{source['id']}] "
                                f"{source['document_name']}**"
                            )
                            st.caption(
                                f"{source['source_label']} · "
                                f"Páginas: {source['pages']}"
                            )
                            st.text(source["text"])
                            st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as exc:
                st.error(
                    "Não foi possível gerar a resposta. "
                    "Verifique o Ollama, o modelo configurado e os "
                    "recursos de busca."
                )
                with st.expander("Detalhes técnicos"):
                    st.exception(exc)
