from multishell.autologin import _build_claude_local_callback_url, _extract_claude_listen_port


def test_extract_claude_listen_port_from_ss_output() -> None:
    ss_output = """
State  Recv-Q Send-Q Local Address:Port  Peer Address:PortProcess
LISTEN 0      511        127.0.0.1:40935      0.0.0.0:*    users:(("claude",pid=12345,fd=14))
"""
    assert _extract_claude_listen_port(ss_output, 12345) == 40935
    assert _extract_claude_listen_port(ss_output, 99999) is None


def test_build_claude_local_callback_url_rewrites_redirect_uri() -> None:
    manual_url = (
        "https://claude.ai/oauth/authorize"
        "?code=true"
        "&client_id=test-client"
        "&response_type=code"
        "&redirect_uri=https%3A%2F%2Fplatform.claude.com%2Foauth%2Fcode%2Fcallback"
        "&scope=user%3Aprofile"
        "&state=test-state"
    )
    rewritten = _build_claude_local_callback_url(manual_url, 40935)
    assert "redirect_uri=http%3A%2F%2Flocalhost%3A40935%2Fcallback" in rewritten
    assert "client_id=test-client" in rewritten
    assert "state=test-state" in rewritten
