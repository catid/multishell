from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .accounts import (
    ACCOUNTS_ENV_VAR,
    PROVIDER_ANTHROPIC,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
    AccountRecord,
    legacy_accounts_present,
    load_accounts_from_env,
    placeholder_accounts,
)
from .envfile import parse_env_value


MODEL = "gpt-5.4"
MODEL_REASONING_EFFORT = "high"
SPARK_MODEL = "gpt-5.3-spark"
SPARK_REASONING_EFFORT = "xhigh"
CLAUDE_MODEL = "claude-opus-4-6"
CLAUDE_REASONING_EFFORT = "high"
MISSING_ENV_PREFIX = "<missing:"
ENV_FILE_ENV_VAR = "MULTISHELL_ENV_FILE"
STATE_ROOT_ENV_VAR = "MULTISHELL_STATE_ROOT"
WORKSPACE_ROOT_ENV_VAR = "MULTISHELL_WORKSPACE_ROOT"
ACCOUNT_CAPACITY_MULTIPLE_ENV_VAR = "MULTISHELL_ACCOUNT_CAPACITY_MULTIPLE"
DEFAULT_DOTENV_TEMPLATE = """# Multishell stores its Google account inventory in this file.
# Use `multishell login` to add or remove accounts and toggle OpenAI, Anthropic, and Gemini access.
MULTISHELL_ACCOUNTS="[]"
"""

_CODEX_PERSONALITIES = (
    (
        "You are a fast implementation specialist. Bias toward shipping the first correct cut quickly, "
        "then tightening rough edges.",
        2,
    ),
    (
        "You are a cautious systems engineer. Bias toward reliability, state management, failure handling, "
        "and operational clarity.",
        3,
    ),
    (
        "You are a product-minded UI engineer. Bias toward terminal UX quality, visual clarity, "
        "and interaction polish.",
        4,
    ),
    (
        "You are a debugging and integration closer. Bias toward verifying joins between pieces, "
        "removing hidden assumptions, and finishing work.",
        5,
    ),
)

_CLAUDE_PERSONALITIES = (
    (
        "You are a creative implementation and review partner. Bias toward diverse candidate code, "
        "novel approaches, and high-signal review comments.",
        2,
    ),
    (
        "You are a systems-minded creative reviewer. Bias toward unusual but viable designs, risk spotting, "
        "and code review from a different model family.",
        3,
    ),
    (
        "You are a product and UX ideation partner. Bias toward creative interface exploration, terminal UX alternatives, "
        "and high-leverage review.",
        4,
    ),
    (
        "You are an integration and review closer with a creative bent. Bias toward diverse bug-hunting, "
        "cross-checking, and alternative code paths.",
        5,
    ),
)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    account_email: str
    role: str
    personality: str
    accent_color: int
    engine: str = "codex"
    account_key: str = ""

    def __post_init__(self) -> None:
        if not self.account_key:
            object.__setattr__(self, "account_key", self.name)


@dataclass(frozen=True)
class ProviderAccountSpec:
    name: str
    account_key: str
    account_email: str
    provider: str
    accent_color: int


def app_root() -> Path:
    return Path(__file__).resolve().parent.parent


def dotenv_template_text() -> str:
    return DEFAULT_DOTENV_TEMPLATE


def state_root() -> Path:
    override = os.environ.get(STATE_ROOT_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".multishell"


def workspace_root() -> Path:
    override = os.environ.get(WORKSPACE_ROOT_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path.cwd().resolve()


def dotenv_path() -> Path:
    override = os.environ.get(ENV_FILE_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()
    return state_root() / ".env"


def _load_dotenv() -> None:
    target = dotenv_path()
    if not target.exists():
        return
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        os.environ.setdefault(key, parse_env_value(value))


def required_env(name: str) -> str:
    return os.environ.get(name, f"<missing:{name}>")


_load_dotenv()


def _initial_account_records() -> list[AccountRecord]:
    records = load_accounts_from_env(os.environ)
    if records:
        return records
    if ACCOUNTS_ENV_VAR in os.environ or legacy_accounts_present(os.environ):
        return records
    return placeholder_accounts()


ACCOUNT_RECORDS = _initial_account_records()
ACCOUNT_RECORDS_BY_KEY = {account.key: account for account in ACCOUNT_RECORDS}

_OPENAI_ACCOUNTS = [account for account in ACCOUNT_RECORDS if account.uses(PROVIDER_OPENAI)]
_ANTHROPIC_ACCOUNTS = [account for account in ACCOUNT_RECORDS if account.uses(PROVIDER_ANTHROPIC)]
_GEMINI_ACCOUNTS = [account for account in ACCOUNT_RECORDS if account.uses(PROVIDER_GEMINI)]


def _missing_account_value(label: str) -> str:
    return f"<missing:{label}>"


def _cycled(values: tuple[tuple[str, int], ...], index: int) -> tuple[str, int]:
    return values[(index - 1) % len(values)]


def _dynamic_account_value(provider: str) -> str:
    return f"<dynamic:{provider}>"


def account_capacity_multiple() -> int:
    raw = os.environ.get(ACCOUNT_CAPACITY_MULTIPLE_ENV_VAR, "").strip()
    try:
        value = int(raw or "2")
    except ValueError:
        value = 2
    return max(1, min(8, value))


def _manager_account() -> AccountRecord | None:
    if _OPENAI_ACCOUNTS:
        return _OPENAI_ACCOUNTS[0]
    return None


_manager_account_record = _manager_account()
MANAGER_SPEC = AgentSpec(
    name="manager",
    account_key=_manager_account_record.key if _manager_account_record is not None else "manager",
    account_email=(
        _manager_account_record.email if _manager_account_record is not None and _manager_account_record.email else _missing_account_value("manager-email")
    ),
    role="manager",
    personality=(
        "You are the delegation manager. You do not directly modify files. "
        "You supervise the workers, break tasks into concrete assignments, "
        "request status, remove blockers, and keep the user informed via tools."
    ),
    accent_color=1,
)

WORKER_SPECS = [
    AgentSpec(
        name=f"worker-{index}",
        account_key=account.key,
        account_email=account.email or _missing_account_value(f"{account.key}-email"),
        role="worker",
        personality=_cycled(_CODEX_PERSONALITIES, index)[0],
        accent_color=_cycled(_CODEX_PERSONALITIES, index)[1],
    )
    for index, account in enumerate(_OPENAI_ACCOUNTS[1:], start=1)
]

CLAUDE_WORKER_SPECS = [
    AgentSpec(
        name=f"claude-worker-{index}",
        account_key=account.key,
        account_email=account.email or _missing_account_value(f"{account.key}-email"),
        role="claude-worker",
        personality=_cycled(_CLAUDE_PERSONALITIES, index)[0],
        accent_color=_cycled(_CLAUDE_PERSONALITIES, index)[1],
        engine="claude",
    )
    for index, account in enumerate(_ANTHROPIC_ACCOUNTS, start=1)
]

GEMINI_ACCOUNT_SPECS = [
    ProviderAccountSpec(
        name=f"gemini-account-{index}",
        account_key=account.key,
        account_email=account.email or _missing_account_value(f"{account.key}-email"),
        provider="gemini_deepthink",
        accent_color=3,
    )
    for index, account in enumerate(_GEMINI_ACCOUNTS, start=1)
]


def codex_account_specs() -> list[ProviderAccountSpec]:
    return [
        ProviderAccountSpec(
            name=f"codex-account-{index}",
            account_key=account.key,
            account_email=account.email or _missing_account_value(f"{account.key}-email"),
            provider=PROVIDER_OPENAI,
            accent_color=_cycled(_CODEX_PERSONALITIES, index)[1],
        )
        for index, account in enumerate(_OPENAI_ACCOUNTS, start=1)
    ]


def claude_account_specs() -> list[ProviderAccountSpec]:
    return [
        ProviderAccountSpec(
            name=f"claude-account-{index}",
            account_key=account.key,
            account_email=account.email or _missing_account_value(f"{account.key}-email"),
            provider=PROVIDER_ANTHROPIC,
            accent_color=_cycled(_CLAUDE_PERSONALITIES, index)[1],
        )
        for index, account in enumerate(_ANTHROPIC_ACCOUNTS, start=1)
    ]


def runtime_codex_worker_specs() -> list[AgentSpec]:
    slot_count = max(0, len(_OPENAI_ACCOUNTS) * account_capacity_multiple() - 1)
    return [
        AgentSpec(
            name=f"worker-{index}",
            account_key=f"worker-slot-{index}",
            account_email=_dynamic_account_value(PROVIDER_OPENAI),
            role="worker",
            personality=_cycled(_CODEX_PERSONALITIES, index)[0],
            accent_color=_cycled(_CODEX_PERSONALITIES, index)[1],
        )
        for index in range(1, slot_count + 1)
    ]


def runtime_claude_worker_specs() -> list[AgentSpec]:
    slot_count = len(_ANTHROPIC_ACCOUNTS) * account_capacity_multiple()
    return [
        AgentSpec(
            name=f"claude-worker-{index}",
            account_key=f"claude-worker-slot-{index}",
            account_email=_dynamic_account_value(PROVIDER_ANTHROPIC),
            role="claude-worker",
            personality=_cycled(_CLAUDE_PERSONALITIES, index)[0],
            accent_color=_cycled(_CLAUDE_PERSONALITIES, index)[1],
            engine="claude",
        )
        for index in range(1, slot_count + 1)
    ]


def manager_workspace_root() -> Path:
    path = state_root() / "workspaces" / "manager"
    path.mkdir(parents=True, exist_ok=True)
    return path


def socket_path() -> Path:
    return state_root() / "control.sock"


def all_agent_specs() -> list[AgentSpec]:
    return [MANAGER_SPEC, *WORKER_SPECS, *CLAUDE_WORKER_SPECS]


def all_codex_specs() -> list[AgentSpec]:
    return [MANAGER_SPEC, *WORKER_SPECS]


def spec_by_name(agent_name: str) -> AgentSpec:
    for spec in all_agent_specs():
        if spec.name == agent_name:
            return spec
    raise KeyError(agent_name)


def maybe_spec_by_name(agent_name: str) -> AgentSpec | None:
    try:
        return spec_by_name(agent_name)
    except KeyError:
        return None


def account_by_key(account_key: str) -> AccountRecord:
    if account_key in ACCOUNT_RECORDS_BY_KEY:
        return ACCOUNT_RECORDS_BY_KEY[account_key]
    raise KeyError(account_key)


def account_for_agent(agent_name: str) -> AccountRecord:
    return account_by_key(spec_by_name(agent_name).account_key)


def gemini_account_specs() -> list[ProviderAccountSpec]:
    return list(GEMINI_ACCOUNT_SPECS)


def default_gemini_account_name() -> str | None:
    if not GEMINI_ACCOUNT_SPECS:
        return None
    return GEMINI_ACCOUNT_SPECS[0].name


def gemini_account_by_name(name: str) -> ProviderAccountSpec:
    for spec in GEMINI_ACCOUNT_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(name)


def provider_account_for_name(name: str) -> AccountRecord:
    return account_by_key(gemini_account_by_name(name).account_key)


def home_owner_name(agent_name: str) -> str:
    if agent_name.endswith("-spark"):
        return home_owner_name(agent_name[: -len("-spark")])
    return agent_name


def password_env_var(agent_name: str) -> str:
    return account_config_label(credential_source_agent(agent_name), "password")


def email_env_var(agent_name: str) -> str:
    return account_config_label(credential_source_agent(agent_name), "email")


def gemini_email_env_var() -> str:
    return f"{ACCOUNTS_ENV_VAR} [gemini email]"


def gemini_password_env_var() -> str:
    return f"{ACCOUNTS_ENV_VAR} [gemini password]"


def account_config_label(account_key: str, field: str) -> str:
    return f"{ACCOUNTS_ENV_VAR} [{account_key} {field}]"


def credential_source_agent(agent_name: str) -> str:
    spec = maybe_spec_by_name(agent_name)
    if spec is not None:
        return spec.account_key
    if agent_name.endswith("-spark"):
        return credential_source_agent(agent_name[: -len("-spark")])
    return home_owner_name(agent_name)


def spark_agent_name(worker_name: str) -> str:
    return f"{worker_name}-spark"


def is_missing_env_value(value: str) -> bool:
    return value.startswith(MISSING_ENV_PREFIX) and value.endswith(">")


def missing_email_env_vars(agent_names: list[str] | None = None) -> list[str]:
    wanted = set(agent_names) if agent_names else None
    missing: list[str] = []
    seen: set[str] = set()
    for spec in all_agent_specs():
        if wanted is not None and spec.name not in wanted:
            continue
        if is_missing_env_value(spec.account_email):
            label = account_config_label(spec.account_key, "email")
            if label not in seen:
                missing.append(label)
                seen.add(label)
    return missing
