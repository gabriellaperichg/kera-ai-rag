import os

import ollama


MODEL_NAME = os.getenv("KERA_OLLAMA_MODEL", "qwen2.5:7b")


def generate_answer(system_prompt: str, user_prompt: str) -> str:
    """Gera uma resposta usando o modelo local do Ollama."""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.1},
    )

    return response["message"]["content"].strip()