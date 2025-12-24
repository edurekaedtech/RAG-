import os
from dotenv import load_dotenv

load_dotenv()

def get_env(name: str, default: str | None = None) -> str:
    val = os.getenv(name, default)
    if val is None:
        raise RuntimeError(f"Missing environment variable: {name}")
    return val
