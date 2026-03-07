from __future__ import annotations

import pytest

from multishell.__main__ import build_parser


def test_auto_login_requires_agent_or_all() -> None:
    parser = build_parser()
    args = parser.parse_args(["auto-login"])
    with pytest.raises(SystemExit, match="provide an agent name or pass --all"):
        args.func(args)
