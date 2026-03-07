from playwright.sync_api import TimeoutError

from multishell.autologin import (
    _build_claude_local_callback_url,
    _click_first,
    _click_google_and_capture_page,
    _extract_claude_listen_port,
)


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


class _NoPopupExpectation:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        raise TimeoutError("no popup")


class _SameTabGooglePage:
    def __init__(self) -> None:
        self.url = "https://chatgpt.com/"
        self.load_waits = 0

    def expect_popup(self, timeout: int):
        return _NoPopupExpectation()

    def wait_for_timeout(self, _timeout_ms: int) -> None:
        return None

    def wait_for_load_state(self, state: str, timeout: int) -> None:
        self.load_waits += 1


def test_click_google_and_capture_page_waits_for_same_tab_google_handoff(monkeypatch) -> None:
    page = _SameTabGooglePage()

    def fake_click_first(page_obj, selectors):
        page_obj.url = "https://accounts.google.com/v3/signin/identifier"

    monkeypatch.setattr("multishell.autologin._click_first", fake_click_first)
    monkeypatch.setattr("multishell.autologin._has_visible", lambda *_args, **_kwargs: False)

    returned = _click_google_and_capture_page(page)

    assert returned is page
    assert page.url.startswith("https://accounts.google.com/")
    assert page.load_waits == 1


class _Candidate:
    def __init__(self, visible: bool, clicked: list[int], index: int) -> None:
        self.visible = visible
        self.clicked = clicked
        self.index = index
        self.first = self

    def wait_for(self, state: str, timeout: int) -> None:
        if not self.visible:
            raise RuntimeError("hidden")

    def click(self) -> None:
        self.clicked.append(self.index)


class _LocatorList:
    def __init__(self, candidates: list[_Candidate]) -> None:
        self._candidates = candidates
        self.first = candidates[0]

    def count(self) -> int:
        return len(self._candidates)

    def nth(self, index: int) -> _Candidate:
        return self._candidates[index]


class _MultiMatchPage:
    def __init__(self, clicked: list[int]) -> None:
        self.url = "https://chatgpt.com/"
        self.clicked = clicked

    def locator(self, selector: str) -> _LocatorList:
        return _LocatorList([_Candidate(False, self.clicked, 0), _Candidate(True, self.clicked, 1)])


def test_click_first_skips_hidden_match_and_uses_visible_one() -> None:
    clicked: list[int] = []
    page = _MultiMatchPage(clicked)

    _click_first(page, ["button:has-text('Log in')"])

    assert clicked == [1]
