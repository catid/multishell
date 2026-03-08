from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Mapping

from .envfile import DotenvFile


ACCOUNTS_ENV_VAR = "MULTISHELL_ACCOUNTS"
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GEMINI = "gemini"
PROVIDER_ORDER = (PROVIDER_OPENAI, PROVIDER_ANTHROPIC, PROVIDER_GEMINI)
_PROVIDER_SET = frozenset(PROVIDER_ORDER)

_LEGACY_SHARED_SLOTS = ("manager", "worker-1", "worker-2", "worker-3", "worker-4")
_LEGACY_GEMINI_SLOT = "gemini"


@dataclass(frozen=True)
class AccountRecord:
    key: str
    email: str
    password: str
    providers: tuple[str, ...]

    def uses(self, provider: str) -> bool:
        return provider in self.providers


def empty_accounts_value() -> str:
    return "[]"


def load_accounts_from_env(environ: Mapping[str, str]) -> list[AccountRecord]:
    if ACCOUNTS_ENV_VAR in environ:
        return parse_accounts_value(environ.get(ACCOUNTS_ENV_VAR, ""))
    return load_legacy_accounts(environ)


def load_accounts_from_dotenv(env_file: DotenvFile) -> list[AccountRecord]:
    raw = env_file.get(ACCOUNTS_ENV_VAR)
    if raw:
        return parse_accounts_value(raw)
    legacy_values = {key: env_file.get(key) for key in legacy_env_keys()}
    return load_legacy_accounts(legacy_values)


def save_accounts_to_dotenv(env_file: DotenvFile, accounts: list[AccountRecord]) -> None:
    env_file.set(ACCOUNTS_ENV_VAR, serialize_accounts(accounts))
    for key in legacy_env_keys():
        env_file.delete(key)


def parse_accounts_value(raw: str) -> list[AccountRecord]:
    text = raw.strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []

    accounts: list[AccountRecord] = []
    seen_keys: set[str] = set()
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        key = _normalize_key(str(item.get("id") or item.get("key") or f"account-{index}"), fallback=f"account-{index}")
        if key in seen_keys:
            key = _dedupe_key(key, seen_keys)
        seen_keys.add(key)
        email = str(item.get("email") or "").strip()
        password = str(item.get("password") or "")
        providers = _normalize_providers(item)
        accounts.append(AccountRecord(key=key, email=email, password=password, providers=providers))
    return accounts


def serialize_accounts(accounts: list[AccountRecord]) -> str:
    payload = [
        {
            "id": account.key,
            "email": account.email,
            "password": account.password,
            "providers": list(account.providers),
        }
        for account in accounts
    ]
    return json.dumps(payload, separators=(",", ":"))


def next_account_key(accounts: list[AccountRecord]) -> str:
    highest = 0
    for account in accounts:
        match = re.fullmatch(r"account-(\d+)", account.key)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"account-{highest + 1}"


def legacy_accounts_present(environ: Mapping[str, str]) -> bool:
    return any(str(environ.get(key) or "").strip() for key in legacy_env_keys())


def legacy_env_keys() -> list[str]:
    keys: list[str] = []
    for slot in _LEGACY_SHARED_SLOTS:
        keys.extend((_legacy_email_env(slot), _legacy_password_env(slot)))
    keys.extend((_legacy_email_env(_LEGACY_GEMINI_SLOT), _legacy_password_env(_LEGACY_GEMINI_SLOT)))
    return keys


def load_legacy_accounts(environ: Mapping[str, str]) -> list[AccountRecord]:
    accounts: list[AccountRecord] = []
    for slot in _LEGACY_SHARED_SLOTS:
        email = str(environ.get(_legacy_email_env(slot), "") or "").strip()
        password = str(environ.get(_legacy_password_env(slot), "") or "")
        if not email and not password:
            continue
        accounts.append(
            AccountRecord(
                key=_normalize_key(slot, fallback=slot),
                email=email,
                password=password,
                providers=(PROVIDER_OPENAI, PROVIDER_ANTHROPIC),
            )
        )
    gemini_email = str(environ.get(_legacy_email_env(_LEGACY_GEMINI_SLOT), "") or "").strip()
    gemini_password = str(environ.get(_legacy_password_env(_LEGACY_GEMINI_SLOT), "") or "")
    if gemini_email or gemini_password:
        accounts.append(
            AccountRecord(
                key="gemini-account-1",
                email=gemini_email,
                password=gemini_password,
                providers=(PROVIDER_GEMINI,),
            )
        )
    return accounts


def placeholder_accounts() -> list[AccountRecord]:
    accounts: list[AccountRecord] = [
        AccountRecord(key="manager", email="", password="", providers=(PROVIDER_OPENAI, PROVIDER_ANTHROPIC)),
    ]
    for index in range(1, 5):
        accounts.append(
            AccountRecord(
                key=f"worker-{index}",
                email="",
                password="",
                providers=(PROVIDER_OPENAI, PROVIDER_ANTHROPIC),
            )
        )
    accounts.append(AccountRecord(key="gemini-account-1", email="", password="", providers=(PROVIDER_GEMINI,)))
    return accounts


def _normalize_providers(item: dict[str, object]) -> tuple[str, ...]:
    providers: set[str] = set()
    raw_providers = item.get("providers")
    if isinstance(raw_providers, list):
        for provider in raw_providers:
            normalized = str(provider or "").strip().lower()
            if normalized in _PROVIDER_SET:
                providers.add(normalized)
    for provider in PROVIDER_ORDER:
        if _truthy(item.get(provider)) or _truthy(item.get(f"use_{provider}")):
            providers.add(provider)
    return tuple(provider for provider in PROVIDER_ORDER if provider in providers)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value or "").strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _legacy_email_env(slot: str) -> str:
    normalized = slot.upper().replace("-", "_")
    return f"MULTISHELL_{normalized}_EMAIL"


def _legacy_password_env(slot: str) -> str:
    normalized = slot.upper().replace("-", "_")
    return f"MULTISHELL_{normalized}_PASSWORD"


def _normalize_key(raw: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9-]+", "-", raw.strip().lower()).strip("-")
    return normalized or fallback


def _dedupe_key(candidate: str, seen: set[str]) -> str:
    suffix = 2
    while f"{candidate}-{suffix}" in seen:
        suffix += 1
    return f"{candidate}-{suffix}"
