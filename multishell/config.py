from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .envfile import parse_env_value


MODEL = "gpt-5.4"
MODEL_REASONING_EFFORT = "medium"
SPARK_MODEL = "gpt-5.3-spark"
SPARK_REASONING_EFFORT = "xhigh"
# Verified locally with `claude -p --model claude-opus-4-6 ...`.
CLAUDE_MODEL = "claude-opus-4-6"
# `high` is the maximum effort level currently exposed by `claude --help`.
CLAUDE_REASONING_EFFORT = "high"
MISSING_ENV_PREFIX = "<missing:"
ENV_FILE_ENV_VAR = "MULTISHELL_ENV_FILE"
STATE_ROOT_ENV_VAR = "MULTISHELL_STATE_ROOT"
WORKSPACE_ROOT_ENV_VAR = "MULTISHELL_WORKSPACE_ROOT"
DEFAULT_DOTENV_TEMPLATE = """MULTISHELL_MANAGER_EMAIL=manager@example.com
MULTISHELL_MANAGER_PASSWORD="replace-me"
# The manager lane is reused by the Codex manager and `claude-worker-5`.

MULTISHELL_WORKER_1_EMAIL=worker1@example.com
MULTISHELL_WORKER_1_PASSWORD="replace-me"

MULTISHELL_WORKER_2_EMAIL=worker2@example.com
MULTISHELL_WORKER_2_PASSWORD="replace-me"

MULTISHELL_WORKER_3_EMAIL=worker3@example.com
MULTISHELL_WORKER_3_PASSWORD="replace-me"

MULTISHELL_WORKER_4_EMAIL=worker4@example.com
MULTISHELL_WORKER_4_PASSWORD="replace-me"

# Gemini Deep Think uses a separate Google OAuth mapping.
# The current implementation uses the manager/bot account only.
MULTISHELL_GEMINI_EMAIL=bot@example.com
MULTISHELL_GEMINI_PASSWORD="replace-me"
"""


@dataclass(frozen=True)
class AgentSpec:
    name: str
    account_email: str
    role: str
    personality: str
    accent_color: int
    engine: str = "codex"


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


MANAGER_SPEC = AgentSpec(
    name="manager",
    account_email=required_env("MULTISHELL_MANAGER_EMAIL"),
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
        name="worker-1",
        account_email=required_env("MULTISHELL_WORKER_1_EMAIL"),
        role="worker",
        personality=(
            "You are a fast implementation specialist. Bias toward shipping the first correct cut, "
            "then tightening rough edges."
        ),
        accent_color=2,
    ),
    AgentSpec(
        name="worker-2",
        account_email=required_env("MULTISHELL_WORKER_2_EMAIL"),
        role="worker",
        personality=(
            "You are a cautious systems engineer. Bias toward reliability, state management, "
            "failure handling, and operational clarity."
        ),
        accent_color=3,
    ),
    AgentSpec(
        name="worker-3",
        account_email=required_env("MULTISHELL_WORKER_3_EMAIL"),
        role="worker",
        personality=(
            "You are a product-minded UI engineer. Bias toward terminal UX quality, visual clarity, "
            "and interaction polish."
        ),
        accent_color=4,
    ),
    AgentSpec(
        name="worker-4",
        account_email=required_env("MULTISHELL_WORKER_4_EMAIL"),
        role="worker",
        personality=(
            "You are a debugging and integration closer. Bias toward verifying joins between pieces, "
            "removing hidden assumptions, and finishing work."
        ),
        accent_color=5,
    ),
]

CLAUDE_WORKER_SPECS = [
    AgentSpec(
        name="claude-worker-5",
        account_email=MANAGER_SPEC.account_email,
        role="claude-worker",
        personality=(
            "You are a creative cross-model reviewer paired with the manager account. "
            "Bias toward fresh ideas, code review, alternative framings, and different-model perspective."
        ),
        accent_color=6,
        engine="claude",
    ),
    AgentSpec(
        name="claude-worker-1",
        account_email=required_env("MULTISHELL_WORKER_1_EMAIL"),
        role="claude-worker",
        personality=(
            "You are a creative implementation and review partner. Bias toward diverse candidate code, "
            "novel approaches, and high-signal review comments."
        ),
        accent_color=2,
        engine="claude",
    ),
    AgentSpec(
        name="claude-worker-2",
        account_email=required_env("MULTISHELL_WORKER_2_EMAIL"),
        role="claude-worker",
        personality=(
            "You are a systems-minded creative reviewer. Bias toward unusual but viable designs, "
            "risk spotting, and code review from a different model family."
        ),
        accent_color=3,
        engine="claude",
    ),
    AgentSpec(
        name="claude-worker-3",
        account_email=required_env("MULTISHELL_WORKER_3_EMAIL"),
        role="claude-worker",
        personality=(
            "You are a product and UX ideation partner. Bias toward creative interface exploration, "
            "terminal UX alternatives, and high-leverage review."
        ),
        accent_color=4,
        engine="claude",
    ),
    AgentSpec(
        name="claude-worker-4",
        account_email=required_env("MULTISHELL_WORKER_4_EMAIL"),
        role="claude-worker",
        personality=(
            "You are an integration and review closer with a creative bent. Bias toward diverse bug-hunting, "
            "cross-checking, and alternative code paths."
        ),
        accent_color=5,
        engine="claude",
    ),
]

def manager_workspace_root() -> Path:
    path = state_root() / "workspaces" / "manager"
    path.mkdir(parents=True, exist_ok=True)
    return path


def socket_path() -> Path:
    return state_root() / "control.sock"


def all_agent_specs() -> list[AgentSpec]:
    return [MANAGER_SPEC, *WORKER_SPECS, *CLAUDE_WORKER_SPECS]


def spec_by_name(agent_name: str) -> AgentSpec:
    for spec in all_agent_specs():
        if spec.name == agent_name:
            return spec
    raise KeyError(agent_name)


def password_env_var(agent_name: str) -> str:
    normalized = credential_source_agent(agent_name).upper().replace("-", "_")
    return f"MULTISHELL_{normalized}_PASSWORD"


def email_env_var(agent_name: str) -> str:
    normalized = credential_source_agent(agent_name).upper().replace("-", "_")
    return f"MULTISHELL_{normalized}_EMAIL"


def gemini_email_env_var() -> str:
    return "MULTISHELL_GEMINI_EMAIL"


def gemini_password_env_var() -> str:
    return "MULTISHELL_GEMINI_PASSWORD"


def credential_source_agent(agent_name: str) -> str:
    if agent_name == "claude-worker-5":
        return "manager"
    if agent_name.startswith("claude-worker-"):
        return agent_name.removeprefix("claude-")
    if agent_name.endswith("-spark"):
        return agent_name[: -len("-spark")]
    return agent_name


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
            env_name = email_env_var(spec.name)
            if env_name not in seen:
                missing.append(env_name)
                seen.add(env_name)
    return missing
