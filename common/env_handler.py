from __future__ import annotations

import os
import sys

from dotenv import dotenv_values, find_dotenv


class EnvError(Exception):
    pass


# Load .env file values
_env = dotenv_values(find_dotenv())

_DB_MAP = {
    "DB_HOST": "POSTGRES_HOST",
    "DB_PORT": "POSTGRES_PORT",
    "DB_NAME": "POSTGRES_DB",
    "DB_USER": "POSTGRES_USER",
    "DB_PASSWORD": "POSTGRES_PASSWORD"
}


def _lookup_key(key: str) -> str | None:
    # 1. Check direct key in system environment
    val = os.environ.get(key)
    if val is not None and val.strip() != "":
        return val.strip()

    # 2. Check direct key in .env file
    val = _env.get(key)
    if val is not None and val.strip() != "":
        return val.strip()

    # 3. Check alias in system environment
    alias = _DB_MAP.get(key)
    if alias:
        val = os.environ.get(alias)
        if val is not None and val.strip() != "":
            return val.strip()
        
        # 4. Check alias in .env file
        val = _env.get(alias)
        if val is not None and val.strip() != "":
            return val.strip()
            
    return None


def get_env(key: str) -> str:
    value = _lookup_key(key)
    if value is None:
        print(f"\n[ENV ERROR] Missing required variable: {key}")
        print(f"Please check your environment variables and .env file.\n")
        sys.exit(1)
    return value


def get_required_env(*keys: str) -> dict:
    missing = []
    values = {}

    for key in keys:
        value = _lookup_key(key)

        if value is None or value.strip() == "":
            missing.append(key)
        else:
            values[key] = value

    if missing:
        print(f"\n[ENV ERROR] Missing required variables: {', '.join(missing)}")
        print(f"Please check your environment variables and .env file.\n")
        sys.exit(1)

    return values