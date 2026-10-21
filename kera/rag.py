
import time
import re
import threading

from scripts.search_hybrid import search
from kera.llm import generate_answer, GenerationCancelled


SYSTEM_PROMPT = """
Você é o Kera AI, assistente de gestão operacional e comercial de clínicas.
Responda em português brasileiro, com clareza, organização visual e linguagem profissional.

REGRAS DE CONTEÚDO:
- Use os trechos fornecidos como fonte factual. Não invente números, metas ou regras.
- Responda à pergunta com as informações relevantes que estiverem nos trechos.
- Não diga que uma informação está ausente se ela estiver explícita no contexto.
- Quando houver divergência, diferencie documento formal de transcrição e explique a divergência.
- Cite as fontes usando somente os identificadores [S1], [S2] etc. fornecidos no contexto.
- Se faltar apenas parte da resposta, responda o que está comprovado e indique especificamente o que faltou.

REGRAS DE FORMATAÇÃO:
- Comece pela resposta, sem frases introdutórias como "Baseado nas informações fornecidas, aqui está...".
- Use parágrafos curtos, separados por uma linha em branco.
- Quando comparar metas ou indicadores, use uma tabela Markdown real, com cada linha em uma linha separada.
- Tabelas devem ter cabeçalho, linha separadora e uma linha por indicador. Nunca escreva uma tabela inteira dentro de um parágrafo.
- Use títulos curtos em negrito somente quando ajudarem a leitura.
- Não repita abaixo da tabela os mesmos dados que já estão nela.
- Evite conclusões genéricas, repetições e explicações que não foram solicitadas.
- Preserve todos os números e suas associações corretas; não misture metas de etapas diferentes.
"""


def _format_pages(metadata: dict) -> str:
    """Converte os metadados de páginas em texto legível."""
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
    """
    Converte os resultados de search_hybrid.py em fontes normalizadas.

    Formato esperado:
        (chunk_id, {
            "score": ...,
            "result": {
                "text": "...",
                "metadata": {...}
            }
        })
    """
    sources = []

    if not results:
        print("[RAG] A busca não retornou resultados.", flush=True)
        return sources

    for position, entry in enumerate(results, start=1):
        try:
            if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                print(
                    f"[RAG] Resultado {position} em formato inválido.",
                    flush=True,
                )
                continue

            chunk_id, item = entry

            if not isinstance(item, dict):
                print(
                    f"[RAG] Resultado {position} não é um dicionário.",
                    flush=True,
                )
                continue

            # O resultado observado no terminal contém os dados em "result".
            nested = item.get("result") or {}

            if not isinstance(nested, dict):
                nested = {}

            metadata = (
                nested.get("metadata")
                or item.get("metadata")
                or {}
            )

            if not isinstance(metadata, dict):
                metadata = {}

            text = nested.get("text") or item.get("text") or ""

            if not isinstance(text, str):
                text = str(text)

            text = text.strip()

            if not text:
                print(
                    f"[RAG] Resultado sem texto: {chunk_id}",
                    flush=True,
                )
                continue

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
                "chunk_id": str(chunk_id),
                "document_name": str(document_name),
                "source_type": source_type,
                "source_label": source_label,
                "pages": _format_pages(metadata),
                "text": text,
                "score": item.get("score"),
                "rerank_score": item.get("rerank_score"),
                "metadata": metadata,
            })

        except Exception as exc:
            print(
                f"[RAG] Erro ao processar resultado {position}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    print(
        f"[RAG] Resultados recebidos: {len(results)} | "
        f"Fontes válidas: {len(sources)}",
        flush=True,
    )

    return sources


def _build_context(sources: list) -> str:
    """Monta o contexto com identificadores de citação."""
    blocks = []

    for source in sources:
        blocks.append(
            f"[{source['id']}]\n"
            f"Tipo de fonte: {source['source_label']}\n"
            f"Documento: {source['document_name']}\n"
            f"Páginas: {source['pages']}\n"
            f"Trecho:\n{source['text']}"
        )

    return "\n\n---\n\n".join(blocks)


def _build_user_prompt(question: str, context: str) -> str:
    """Monta instruções para uma resposta legível sem alterar a recuperação."""
    return f"""PERGUNTA:
{question}

FONTES DISPONÍVEIS:
{context}

Responda diretamente à pergunta usando somente as fontes acima.
Se a pergunta envolver metas ou indicadores, organize os dados em uma tabela Markdown
com as colunas mais adequadas (por exemplo: Indicador, Meta mínima e Meta ideal).
Escreva cada linha da tabela em uma linha separada e inclua a linha separadora Markdown.
Coloque uma linha em branco antes e depois da tabela.
Associe cada número à etapa correta e cite a fonte na mesma linha quando aplicável.
Se parte da resposta não estiver documentada, indique exatamente qual parte falta.
Não inclua introdução genérica, não repita os dados da tabela em prosa e use parágrafos curtos."""


def _format_answer(answer: str) -> str:
    """Melhora espaçamento e transforma tabelas compactadas em Markdown legível."""
    answer = (answer or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    # Se o modelo inseriu uma tabela no meio de um parágrafo, separa o bloco.
    headers = (
        "Indicador | Meta Mínima",
        "Indicador | Meta mínima",
        "Etapa | Meta Mínima",
        "Etapa | Meta mínima",
    )
    for header in headers:
        idx = answer.find(header)
        if idx > 0:
            answer = answer[:idx].rstrip() + "\n\n" + answer[idx:]
            break

    # Corrige o formato específico de tabela que o modelo costuma compactar.
    # Mantém os dados gerados pelo modelo; apenas reorganiza sua apresentação.
    compact_table = re.search(
        r"(Indicador\s*\|\s*Meta\s*Mínima\s*\|\s*Meta\s*Ideal)(.*)",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if compact_table:
        prefix = answer[:compact_table.start()].rstrip()
        table_tail = compact_table.group(2).strip()
        table_tail = re.sub(
            r"\s+(Leads Recebidos|Agendamentos|Comparecimento|Vendas)\s*\|",
            r"\n\1 |",
            table_tail,
            flags=re.IGNORECASE,
        )
        table = "Indicador | Meta Mínima | Meta Ideal\n|---|---:|---:|\n" + table_tail
        # Evita deixar a frase de conclusão colada ao último valor da tabela.
        table = re.sub(
            r"(\|\s*[^|\n]+)\s+(Essas metas|Esses indicadores|A meta ideal|Esses resultados)\b",
            r"\1\n\n\2",
            table,
            flags=re.IGNORECASE,
        )
        answer = (prefix + "\n\n" if prefix else "") + table

    # Normaliza espaços e adiciona respiro entre parágrafos e tabelas.
    lines = [line.rstrip() for line in answer.split("\n")]
    formatted = []
    for i, line in enumerate(lines):
        current = line.strip()
        is_table = "|" in current
        prev_is_table = bool(formatted and "|" in formatted[-1].strip())
        next_is_table = i + 1 < len(lines) and "|" in lines[i + 1].strip()
        if is_table and formatted and formatted[-1].strip() and not prev_is_table:
            formatted.append("")
        formatted.append(line)
        if is_table and not next_is_table and i + 1 < len(lines) and lines[i + 1].strip():
            formatted.append("")

    answer = "\n".join(formatted)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    return answer.strip()


def answer_question(question: str, resources: dict, cancel_event: threading.Event | None = None) -> dict:
    """Recupera evidências e gera uma resposta fundamentada."""
    question = (question or "").strip()

    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelled("Geração cancelada pelo usuário.")

    if not question:
        return {
            "answer": "Digite uma pergunta.",
            "sources": [],
        }

    print("\n=== KERA RAG ===", flush=True)
    print(f"[RAG] Pergunta: {question}", flush=True)

    # Etapa 1: busca. A busca em si é síncrona; cancelamos logo que ela retornar.
    start_search = time.perf_counter()

    try:
        # O teste mostrou que um trecho relevante pode ficar em 8º lugar.
        # Mantemos até 8 resultados no contexto para não descartá-lo cedo demais.
        results = search(question, resources, top_k=8)
    except Exception as exc:
        print(
            f"[RAG] Erro na busca: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return {
            "answer": (
                "Não foi possível pesquisar a base de conhecimento. "
                "Confira o terminal para ver o erro."
            ),
            "sources": [],
        }

    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelled("Geração cancelada pelo usuário.")

    print(
        f"[RAG] Busca concluída em "
        f"{time.perf_counter() - start_search:.2f}s; "
        f"resultados: {len(results) if results else 0}.",
        flush=True,
    )

    # Etapa 2: normalização das fontes.
    sources = _prepare_sources(results or [])

    if not sources:
        return {
            "answer": (
                "Não encontrei trechos de texto suficientes para "
                "responder com segurança. Tente reformular a pergunta."
            ),
            "sources": [],
        }

    context = _build_context(sources)

    print(
        f"[RAG] Contexto montado: {len(context)} caracteres, "
        f"{len(sources)} fontes.",
        flush=True,
    )

    # Etapa 3: prompt específico para a pergunta.
    user_prompt = _build_user_prompt(question, context)

    # Etapa 4: geração pelo Ollama, em streaming para permitir cancelamento.
    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelled("Geração cancelada pelo usuário.")

    print("[RAG] Enviando contexto ao modelo...", flush=True)
    start_generation = time.perf_counter()

    try:
        answer = generate_answer(SYSTEM_PROMPT, user_prompt, cancel_event=cancel_event)
    except GenerationCancelled:
        raise
    except Exception as exc:
        print(
            f"[RAG] Erro na geração: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return {
            "answer": (
                "Não foi possível gerar a resposta pelo modelo de linguagem. "
                "Confira se o Ollama está funcionando e veja o erro no terminal."
            ),
            "sources": sources,
        }

    elapsed = time.perf_counter() - start_generation

    if not isinstance(answer, str) or not answer.strip():
        print("[RAG] O modelo retornou uma resposta vazia.", flush=True)
        return {
            "answer": (
                "O modelo não retornou uma resposta válida. "
                "Tente novamente."
            ),
            "sources": sources,
        }

    answer = answer.strip()

    print(
        f"[RAG] Geração concluída em {elapsed:.2f}s; "
        f"resposta: {len(answer)} caracteres.",
        flush=True,
    )

    return {
        "answer": answer,
        "sources": sources,
    }
