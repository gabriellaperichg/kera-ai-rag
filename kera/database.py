from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Compatibilidade: se o banco antigo estiver na raiz, ele continua sendo usado.
OLD_DB_PATH = PROJECT_ROOT / "kera_users.db"
DB_PATH = OLD_DB_PATH if OLD_DB_PATH.exists() else DATA_DIR / "kera_users.db"


def db():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with db() as conn:
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


def password_hash(password: str, salt_hex: str | None = None):
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
    return digest.hex(), salt.hex()


def verify_password(password: str, salt_hex: str, expected_hash: str) -> bool:
    actual_hash, _ = password_hash(password, salt_hex)
    return hmac.compare_digest(actual_hash, expected_hash)


def user_count() -> int:
    with db() as conn:
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def create_user(username: str, display_name: str, password: str, role: str = "user"):
    username = username.strip()
    display_name = display_name.strip() or username
    if len(username) < 3:
        raise ValueError("O usuário precisa ter pelo menos 3 caracteres.")
    if len(password) < 10:
        raise ValueError("A senha precisa ter pelo menos 10 caracteres.")
    pw_hash, salt = password_hash(password)
    with db() as conn:
        conn.execute(
            """INSERT INTO users
               (username, display_name, password_hash, password_salt, role, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (username, display_name, pw_hash, salt, role, datetime.now().isoformat(timespec="seconds")),
        )


def authenticate(username: str, password: str):
    with db() as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE",
            (username.strip(),),
        ).fetchone()
    if user and verify_password(password, user["password_salt"], user["password_hash"]):
        return dict(user)
    return None


def new_conversation(user_id: int):
    now = datetime.now().isoformat(timespec="seconds")
    conv = {
        "id": str(uuid.uuid4()), "user_id": user_id, "title": "Nova conversa",
        "created_at": now, "updated_at": now, "messages": []
    }
    with db() as conn:
        conn.execute(
            "INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (conv["id"], user_id, conv["title"], now, now),
        )
    return conv


def load_conversations(user_id: int):
    with db() as conn:
        rows = conn.execute(
            """SELECT id, user_id, title, created_at, updated_at
               FROM conversations WHERE user_id = ? ORDER BY updated_at DESC""",
            (user_id,),
        ).fetchall()
        result = []
        for row in rows:
            msgs = conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
                (row["id"],),
            ).fetchall()
            result.append({
                "id": row["id"], "user_id": row["user_id"], "title": row["title"],
                "created_at": row["created_at"], "updated_at": row["updated_at"],
                "messages": [{"role": m["role"], "content": m["content"]} for m in msgs],
            })
    return result


def save_message(conversation, role: str, content: str):
    now = datetime.now().isoformat(timespec="seconds")
    with db() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation["id"], role, content, now),
        )
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (conversation["title"], now, conversation["id"], conversation["user_id"]),
        )
    conversation["updated_at"] = now


def save_title(conversation):
    now = datetime.now().isoformat(timespec="seconds")
    with db() as conn:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (conversation["title"], now, conversation["id"], conversation["user_id"]),
        )
    conversation["updated_at"] = now


def list_users():
    with db() as conn:
        return conn.execute(
            "SELECT username, display_name, role, created_at FROM users ORDER BY id"
        ).fetchall()


def rename_conversation(user_id: int, conversation_id: str, title: str) -> bool:
    """Renomeia uma conversa pertencente ao usuário informado."""
    title = " ".join(title.split()).strip()
    if not title:
        raise ValueError("O título da conversa não pode ficar vazio.")
    title = title[:100]
    now = datetime.now().isoformat(timespec="seconds")
    with db() as conn:
        cursor = conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (title, now, conversation_id, user_id),
        )
    return cursor.rowcount > 0


def delete_conversation(user_id: int, conversation_id: str) -> bool:
    """Exclui uma conversa do próprio usuário; mensagens são removidas via FK cascade."""
    with db() as conn:
        cursor = conn.execute(
            "DELETE FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        )
    return cursor.rowcount > 0
