
import os
import ollama


MODEL_NAME = os.getenv("KERA_OLLAMA_MODEL", "qwen2.5:3b")


def generate_answer(system_prompt: str, user_prompt: str) -> str:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        options={
            "temperature": 0.0,
            "num_predict": 1200,
            "num_ctx": 4096,
        },
    )

    return response["message"]["content"].strip()
