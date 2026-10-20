
from scripts.search_hybrid import load_search_resources, search

from kera.llm import generate_answer


SYSTEM_PROMPT = """
Você é o Kera AI, assistente de gestão operacional e comercial de clínicas.

REGRAS OBRIGATÓRIAS:

1. Responda em português brasileiro, com clareza e objetividade.
2. Use exclusivamente as evidências presentes no contexto fornecido.
3. Não invente metas, frequências, procedimentos, exceções ou regras.
4. Se o contexto não sustentar uma resposta completa, explique o que
   é possível afirmar e o que não está especificado nas fontes.
5. Se não houver evidência suficiente para responder, admita isso
   explicitamente. Não preencha lacunas com conhecimento geral.
6. Cite as evidências no texto usando os identificadores [S1], [S2]
   etc. Use somente identificadores existentes no contexto.
7. Diferencie documentos formais de transcrições:
   - Documento formal: diretriz ou documentação estratégica.
   - Transcrição: relato prático de uma pessoa entrevistada.
   Não apresente um relato de transcrição como regra formal da empresa.
8. Se duas fontes apresentarem informações diferentes, informe a
   diferença sem escolher uma delas arbitrariamente.
9. Não diga que consultou uma fonte se ela não estiver no contexto.
10. Não invente páginas ou referências.
"""


def _format_pages(metadata: dict) -> str:
    pages = metadata.get("pages")

    if isinstance(pages, (list, tuple)):
        pages = ", ".join(str(page) for page in pages)

    if pages:
        return str(pages)

    start = metadata.get("page_start", metadata.get("page"))
    end = metadata.get("page_end")

    if start is None:
        return "não identificadas"

    if end is not None and str(end) != str(start):
        return f"{start}–{end}"

    return str(start)


def _prepare_sources(results: list) -> list:
    sources = []

    for chunk_id, item in results:
        metadata = item.get("metadata") or {}

        # Compatibilidade com resultados que trazem campos na raiz.
        if not metadata:
            metadata = item

        text = item.get("text", "")
        source_type = metadata.get("source_type", "unknown")

        if source_type == "transcript":
            source_label = "Transcrição"
        elif source_type == "document":
            source_label = "Documento formal"
        else:
            source_label = "Tipo de fonte não identificado"

        document_name = metadata.get(
            "document_name",
            metadata.get("document_id", "Documento não identificado"),
        )

        sources.append({
            "id": f"S{len(sources) + 1}",
            "chunk_id": chunk_id,
            "document_name": document_name,
            "source_type": source_type,
            "source_label": source_label,
            "pages": _format_pages(metadata),
            "text": text,
            "rerank_score": item.get("rerank_score"),
        })

    return sources


def _build_context(sources: list) -> str:
    blocks = []

    for source in sources:
        blocks.append(
            f"[{source['id']}]\n"
            f"Tipo: {source['source_label']}\n"
            f"Documento: {source['document_name']}\n"
            f"Páginas: {source['pages']}\n"
            f"Trecho:\n{source['text']}"
        )

    return "\n\n---\n\n".join(blocks)


def answer_question(question: str, resources: dict) -> dict:
    question = question.strip()

    if not question:
        raise ValueError("Digite uma pergunta.")

    results = search(question, resources)
    sources = _prepare_sources(results)

    if not sources or not any(source["text"].strip() for source in sources):
        return {
            "answer": (
                "Não encontrei trechos de texto suficientes para "
                "responder com segurança. Tente reformular a pergunta."
            ),
            "sources": [],
        }

    context = _build_context(sources)

    user_prompt = f"""
PERGUNTA:
{question}

CONTEXTO RECUPERADO:
{context}

INSTRUÇÃO FINAL:
Responda à pergunta usando somente o contexto recuperado.
Inclua citações [S1], [S2] etc. junto às afirmações que sustentam.
Se os trechos não forem suficientes, diga isso claramente.
"""

    answer = generate_answer(SYSTEM_PROMPT, user_prompt)

    return {
        "answer": answer,
        "sources": sources,
    }
