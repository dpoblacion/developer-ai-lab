"""Minimal .env loader for local secrets (no dependency).

`.env` holds local-only secrets (RUNPOD_API_KEY, SSH_KEY_PATH) and is gitignored.
``parse_dotenv`` is pure; ``load_dotenv`` reads the file into os.environ without
overriding values already set explicitly in the environment.
"""

import os


def parse_dotenv(text):
    """Parse KEY=VALUE lines into a dict. Ignores blanks and # comments."""
    env = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        env[key.strip()] = value
    return env


def apply_dotenv(text, target):
    """Apply parsed values to ``target`` (a dict) without overriding existing keys."""
    for key, value in parse_dotenv(text).items():
        target.setdefault(key, value)


def load_dotenv(path=".env"):
    """Load a .env file into os.environ if it exists (existing env vars win)."""
    if os.path.exists(path):
        with open(path) as f:
            apply_dotenv(f.read(), os.environ)
