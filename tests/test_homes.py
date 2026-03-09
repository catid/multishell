from multishell.config import SPARK_MODEL, SPARK_REASONING_EFFORT
from multishell.homes import account_home, agent_home, claude_home, ensure_agent_home, ensure_claude_home, config_path


def test_manager_config_contains_model() -> None:
    ensure_agent_home("manager", manager_bridge_command=["python3", "-m", "multishell.mcp_bridge", "--socket", "/tmp/x"])
    config = config_path("manager").read_text(encoding="utf-8")
    assert 'model = "gpt-5.4"' in config
    assert 'model_reasoning_effort = "high"' in config
    assert "[mcp_servers.multishell]" in config


def test_claude_home_is_isolated_from_account_home() -> None:
    assert ensure_claude_home("claude-worker-1") == claude_home("claude-worker-1")
    assert claude_home("claude-worker-1") != account_home("manager")
    assert claude_home("claude-worker-2") == agent_home("claude-worker-2")


def test_spark_home_is_isolated_from_worker_home() -> None:
    worker_bridge = ["python3", "-m", "multishell.mcp_bridge", "--socket", "/tmp/x", "--role", "worker", "--agent", "worker-1"]
    ensure_agent_home("worker-1", mcp_bridge_command=worker_bridge)
    ensure_agent_home(
        "worker-1-spark",
        model=SPARK_MODEL,
        reasoning_effort=SPARK_REASONING_EFFORT,
    )

    worker_config = config_path("worker-1").read_text(encoding="utf-8")
    spark_config = config_path("worker-1-spark").read_text(encoding="utf-8")

    assert agent_home("worker-1-spark") != agent_home("worker-1")
    assert 'model = "gpt-5.4"' in worker_config
    assert "[mcp_servers.multishell]" in worker_config
    assert f'model = "{SPARK_MODEL}"' in spark_config
    assert "[mcp_servers.multishell]" not in spark_config
