from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DotenvLine:
    raw: str
    key: str | None = None
    value: str | None = None

    @property
    def is_kv(self) -> bool:
        return self.key is not None


def parse_env_value(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    if len(text) >= 2 and text[0] == text[-1] == '"':
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return text[1:-1]
        return value if isinstance(value, str) else str(value)
    if len(text) >= 2 and text[0] == text[-1] == "'":
        return text[1:-1]
    return text


def format_env_value(value: str) -> str:
    return json.dumps(value)


def parse_dotenv(text: str) -> list[DotenvLine]:
    lines: list[DotenvLine] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in raw:
            lines.append(DotenvLine(raw=raw))
            continue
        key, value = raw.split("=", 1)
        lines.append(DotenvLine(raw=raw, key=key.strip(), value=parse_env_value(value)))
    return lines


def ensure_dotenv_file(path: Path, template_text: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    content = template_text if template_text.endswith("\n") else template_text + "\n"
    path.write_text(content, encoding="utf-8")


class DotenvFile:
    def __init__(self, path: Path, lines: list[DotenvLine]) -> None:
        self.path = path
        self._lines = lines

    @classmethod
    def load(cls, path: Path, template_text: str) -> DotenvFile:
        ensure_dotenv_file(path, template_text)
        text = path.read_text(encoding="utf-8")
        return cls(path, parse_dotenv(text))

    def get(self, key: str, default: str = "") -> str:
        for line in self._lines:
            if line.key == key:
                return line.value or ""
        return default

    def set(self, key: str, value: str) -> None:
        for line in self._lines:
            if line.key == key:
                line.value = value
                return
        self._lines.append(DotenvLine(raw="", key=key, value=value))

    def save(self) -> None:
        rendered: list[str] = []
        seen: set[str] = set()
        for line in self._lines:
            if line.key is None:
                rendered.append(line.raw)
                continue
            if line.key in seen:
                continue
            rendered.append(f"{line.key}={format_env_value(line.value or '')}")
            seen.add(line.key)
        content = "\n".join(rendered).rstrip("\n") + "\n"
        self.path.write_text(content, encoding="utf-8")
