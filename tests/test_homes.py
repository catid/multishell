from multishell.homes import agent_home, claude_home, ensure_agent_home, ensure_claude_home, config_path


def test_manager_config_contains_model() -> None:
    ensure_agent_home("manager", manager_bridge_command=["python3", "-m", "multishell.mcp_bridge", "--socket", "/tmp/x"])
    config = config_path("manager").read_text(encoding="utf-8")
    assert 'model = "gpt-5.4"' in config
    assert 'model_reasoning_effort = "medium"' in config
    assert "[mcp_servers.multishell]" in config


def test_claude_home_shares_account_slot_with_codex_worker() -> None:
    assert ensure_claude_home("claude-worker-5") == agent_home("manager")
    assert claude_home("claude-worker-1") == agent_home("worker-1")
