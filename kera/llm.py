import os
import threading
import ollama


MODEL_NAME = os.getenv("KERA_OLLAMA_MODEL", "qwen2.5:3b")


class GenerationCancelled(Exception):
    """Raised when a generation is cancelled by the user."""


def generate_answer(
    system_prompt: str,
    user_prompt: str,
    cancel_event: threading.Event | None = None,
) -> str:
    """
    Gera a resposta em streaming para permitir verificar o pedido de cancelamento.
    Ao cancelar, encerra a iteração do stream e não retorna resposta parcial.
    """
    client = ollama.Client()
    stream = client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": 0.0,
            "num_predict": 1200,
            "num_ctx": 4096,
        },
        stream=True,
    )

    chunks = []
    try:
        for chunk in stream:
            if cancel_event is not None and cancel_event.is_set():
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
                raise GenerationCancelled("Geração cancelada pelo usuário.")

            message = chunk.get("message") or {}
            content = message.get("content", "")
            if content:
                chunks.append(content)

        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled("Geração cancelada pelo usuário.")

    finally:
        # Closing the iterator helps release the HTTP stream when cancellation occurs.
        if cancel_event is not None and cancel_event.is_set():
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    return "".join(chunks).strip()
