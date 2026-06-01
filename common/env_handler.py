import sys

from dotenv import dotenv_values, find_dotenv


class EnvError(Exception):
    pass


# Load ONLY .env file (no Windows/system env)
_env = dotenv_values(find_dotenv())


def get_env(key: str) -> str:
    value = _env.get(key)
    if value is None or value.strip() == "":
        print(f"\n[ENV ERROR] Missing required variable: {key}")
        print(f"Please check your .env file.\n")
        sys.exit(1)
    return value.strip()


def get_required_env(*keys: str) -> dict:
    missing = []
    values = {}

    for key in keys:
        value = _env.get(key)

        if value is None or value.strip() == "":
            missing.append(key)
        else:
            values[key] = value.strip()

    if missing:
        print(f"\n[ENV ERROR] Missing required variables: {', '.join(missing)}")
        print(f"Please check your .env file.\n")
        sys.exit(1)

    return values