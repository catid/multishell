from __future__ import annotations

from pathlib import Path

from multishell.envfile import DotenvFile, format_env_value, parse_dotenv, parse_env_value


def test_parse_env_value_supports_json_strings() -> None:
    assert parse_env_value('"pa\\\"ss"') == 'pa"ss'


def test_parse_dotenv_reads_key_values_and_comments() -> None:
    lines = parse_dotenv("# comment\nKEY=value\nEMPTY=\n")

    assert [line.key for line in lines] == [None, "KEY", "EMPTY"]
    assert lines[1].value == "value"
    assert lines[2].value == ""


def test_dotenv_file_updates_existing_and_appends_missing(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("# header\nEXISTING=\"old\"\n", encoding="utf-8")

    env_file = DotenvFile.load(env_path, "EXISTING=\"old\"\n")
    updated_value = 'new"value'
    env_file.set("EXISTING", updated_value)
    env_file.set("ADDED", "fresh")
    env_file.save()

    saved = env_path.read_text(encoding="utf-8")
    assert "# header" in saved
    assert f"EXISTING={format_env_value(updated_value)}" in saved
    assert 'ADDED="fresh"' in saved
