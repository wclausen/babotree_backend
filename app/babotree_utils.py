import os
from pathlib import Path

LOCAL_SECRET_FILE = str(Path(__file__).parent.parent) + "/.env"


def get_secret(key: str) -> str:
    env_secret = os.environ.get(key)
    if env_secret is not None:
        return env_secret
    else:
        return _get_local_secret(key)


def _get_local_secret(key: str) -> str:
    with open(LOCAL_SECRET_FILE, "r") as local_secrets_file:
        lines = local_secrets_file.readlines()
        for line in lines:
            if line.split("=")[0] == key:
                return line.split("=")[1].strip()