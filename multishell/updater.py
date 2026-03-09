from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import state_root
from .runtime import install_root


DEFAULT_GITHUB_REPO = "catid/multishell"
ENV_GITHUB_REPO = "MULTISHELL_GITHUB_REPO"
ENV_AUTO_UPDATE = "MULTISHELL_AUTO_UPDATE"
AUTO_UPDATE_MODE_CHECK = "check"
AUTO_UPDATE_MODE_APPLY = "apply"
AUTO_UPDATE_MODE_OFF = "off"
STARTUP_CHECK_INTERVAL_SECONDS = 24 * 60 * 60


class UpdaterError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReleaseInfo:
    tag_name: str
    version: str
    tarball_url: str
    html_url: str


@dataclass(frozen=True)
class StartupUpdateResult:
    message: str | None = None
    restart_python: Path | None = None
    available_release: ReleaseInfo | None = None


def configured_github_repo() -> str:
    return os.environ.get(ENV_GITHUB_REPO, "").strip() or DEFAULT_GITHUB_REPO


def current_version() -> str:
    try:
        return importlib.metadata.version("multishell")
    except importlib.metadata.PackageNotFoundError:
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if pyproject.exists():
            for raw_line in pyproject.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if line.startswith("version = "):
                    return line.split('"', 2)[1]
        raise UpdaterError("unable to determine current multishell version")


def check_for_update(*, repo: str | None = None, current: str | None = None) -> ReleaseInfo | None:
    current_version_text = current or current_version()
    latest = latest_release(repo=repo)
    if latest is None:
        return None
    if not _is_newer_version(latest.version, current_version_text):
        return None
    return latest


def latest_release(*, repo: str | None = None) -> ReleaseInfo | None:
    selected_repo = repo or configured_github_repo()
    payload = _fetch_json(
        f"https://api.github.com/repos/{selected_repo}/tags?per_page=100",
    )
    if not isinstance(payload, list):
        raise UpdaterError(f"unexpected GitHub tags payload for {selected_repo}")
    releases = [_release_from_tag_payload(selected_repo, item) for item in payload]
    releases = [release for release in releases if release is not None]
    if not releases:
        return None
    return max(releases, key=lambda release: _version_key(release.version))


def installed_release_versions(target_install_root: Path | None = None) -> list[str]:
    releases_dir = _releases_dir(target_install_root)
    if not releases_dir.exists():
        return []
    versions = [entry.name for entry in releases_dir.iterdir() if entry.is_dir()]
    return sorted(versions, key=_version_key)


def rollback_to_version(
    *,
    version: str | None = None,
    target_install_root: Path | None = None,
    target_bin_dir: Path | None = None,
) -> str:
    versions = installed_release_versions(target_install_root)
    if not versions:
        raise UpdaterError("no installed releases are available to roll back to")
    current = current_release_version(target_install_root)
    target_version = version or _default_rollback_target(versions, current)
    if target_version is None:
        raise UpdaterError("no previous release is available to roll back to")
    if target_version not in versions:
        raise UpdaterError(f"release {target_version} is not installed")
    _activate_installed_release(target_version, target_install_root=target_install_root, target_bin_dir=target_bin_dir)
    return target_version


def current_release_version(target_install_root: Path | None = None) -> str | None:
    current_link = _current_link(target_install_root)
    if not current_link.exists():
        return None
    try:
        return current_link.resolve().name
    except OSError as exc:
        raise UpdaterError(f"failed to resolve current release link: {exc}") from exc


def install_latest_release(
    *,
    repo: str | None = None,
    python_bin: str | Path | None = None,
    target_install_root: Path | None = None,
    target_bin_dir: Path | None = None,
) -> ReleaseInfo:
    release = latest_release(repo=repo)
    if release is None:
        raise UpdaterError("no tagged releases are available")
    install_release(
        release,
        python_bin=python_bin,
        target_install_root=target_install_root,
        target_bin_dir=target_bin_dir,
    )
    return release


def install_release(
    release: ReleaseInfo,
    *,
    python_bin: str | Path | None = None,
    target_install_root: Path | None = None,
    target_bin_dir: Path | None = None,
) -> Path:
    selected_install_root = Path(target_install_root or install_root()).expanduser()
    selected_bin_dir = Path(target_bin_dir or (Path.home() / ".local" / "bin")).expanduser()
    selected_python = Path(python_bin or sys.executable)

    selected_install_root.mkdir(parents=True, exist_ok=True)
    selected_bin_dir.mkdir(parents=True, exist_ok=True)

    release_dir = _release_dir(release.version, selected_install_root)
    if _release_dir_ready(release_dir):
        _activate_release_dir(release_dir, target_install_root=selected_install_root, target_bin_dir=selected_bin_dir)
        return release_dir

    releases_dir = _releases_dir(selected_install_root)
    releases_dir.mkdir(parents=True, exist_ok=True)

    staging_dir = releases_dir / f".{release.version}.stage-{uuid.uuid4().hex[:8]}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    with tempfile.TemporaryDirectory(prefix="multishell-update-") as tmp_root:
        tmp_path = Path(tmp_root)
        source_dir = _download_and_extract_release(release, tmp_path)
        app_dir = staging_dir / "app"
        venv_dir = staging_dir / "venv"
        shutil.copytree(source_dir, app_dir)
        _build_release_venv(app_dir, venv_dir, python_bin=selected_python)

    if release_dir.exists():
        shutil.rmtree(release_dir)
    staging_dir.rename(release_dir)
    _activate_release_dir(release_dir, target_install_root=selected_install_root, target_bin_dir=selected_bin_dir)
    return release_dir


def check_startup_update(*, repo: str | None = None) -> StartupUpdateResult:
    mode = _auto_update_mode()
    if mode == AUTO_UPDATE_MODE_OFF:
        return StartupUpdateResult()
    if not _running_from_managed_install():
        return StartupUpdateResult()
    if not _should_check_for_update():
        return StartupUpdateResult()
    try:
        update = check_for_update(repo=repo)
    except UpdaterError:
        _record_update_check()
        return StartupUpdateResult()

    _record_update_check()
    if update is None:
        return StartupUpdateResult()
    if mode == AUTO_UPDATE_MODE_APPLY:
        install_release(update)
        restart_python = current_release_python()
        if restart_python is None:
            return StartupUpdateResult(message=f"updated multishell to {update.tag_name}")
        return StartupUpdateResult(
            message=f"updated multishell to {update.tag_name}; restarting",
            restart_python=restart_python,
        )
    return StartupUpdateResult(
        message=(
            f"update available: {current_version()} -> {update.version} "
            f"({update.tag_name}). Run `multishell update`."
        ),
        available_release=update,
    )


def current_release_python(target_install_root: Path | None = None) -> Path | None:
    current_link = _current_link(target_install_root)
    python_path = current_link / "venv" / "bin" / "python"
    if python_path.exists():
        return python_path
    legacy_python = Path(target_install_root or install_root()).expanduser() / "venv" / "bin" / "python"
    if legacy_python.exists():
        return legacy_python
    return None


def _download_and_extract_release(release: ReleaseInfo, temp_root: Path) -> Path:
    tarball_path = temp_root / "multishell-release.tar.gz"
    _download_file(release.tarball_url, tarball_path)
    extract_root = temp_root / "src"
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball_path, "r:gz") as archive:
        archive.extractall(extract_root)
    extracted_dirs = [entry for entry in extract_root.iterdir() if entry.is_dir()]
    if not extracted_dirs:
        raise UpdaterError(f"downloaded archive for {release.tag_name} did not contain a source directory")
    return extracted_dirs[0]


def _download_file(url: str, target: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "multishell-updater",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            target.write_bytes(response.read())
    except urllib.error.URLError as exc:
        raise UpdaterError(f"failed to download {url}: {exc}") from exc


def _fetch_json(url: str) -> object:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "multishell-updater",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise UpdaterError(f"failed to query {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise UpdaterError(f"invalid JSON from {url}: {exc}") from exc


def _release_from_tag_payload(repo: str, payload: object) -> ReleaseInfo | None:
    if not isinstance(payload, dict):
        return None
    tag_name = str(payload.get("name") or "").strip()
    version = _normalize_version_tag(tag_name)
    if not tag_name or version is None:
        return None
    return ReleaseInfo(
        tag_name=tag_name,
        version=version,
        tarball_url=f"https://codeload.github.com/{repo}/tar.gz/{tag_name}",
        html_url=f"https://github.com/{repo}/releases/tag/{tag_name}",
    )


def _normalize_version_tag(tag_name: str) -> str | None:
    normalized = tag_name.strip()
    if normalized.startswith("refs/tags/"):
        normalized = normalized[len("refs/tags/") :]
    if normalized.startswith("v"):
        normalized = normalized[1:]
    if _parse_version(normalized) is None:
        return None
    return normalized


def _parse_version(value: str) -> tuple[int, int, int] | None:
    parts = value.split(".", 2)
    if len(parts) != 3:
        return None
    parsed: list[int] = []
    for index, part in enumerate(parts):
        digits = []
        for char in part:
            if char.isdigit():
                digits.append(char)
                continue
            if index == 2 and digits:
                break
            return None
        if not digits:
            return None
        parsed.append(int("".join(digits)))
    return parsed[0], parsed[1], parsed[2]


def _version_key(value: str) -> tuple[int, int, int]:
    parsed = _parse_version(value)
    if parsed is None:
        raise UpdaterError(f"unsupported release version format: {value}")
    return parsed


def _is_newer_version(candidate: str, current: str) -> bool:
    return _version_key(candidate) > _version_key(current)


def _build_release_venv(app_dir: Path, venv_dir: Path, *, python_bin: Path) -> None:
    subprocess.run([str(python_bin), "-m", "venv", str(venv_dir)], check=True)
    release_python = venv_dir / "bin" / "python"
    subprocess.run([str(release_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([str(release_python), "-m", "pip", "install", str(app_dir)], check=True)
    subprocess.run([str(release_python), "-m", "multishell", "--help"], check=True, capture_output=True, text=True)


def _activate_installed_release(
    version: str,
    *,
    target_install_root: Path | None = None,
    target_bin_dir: Path | None = None,
) -> None:
    release_dir = _release_dir(version, target_install_root)
    if not _release_dir_ready(release_dir):
        raise UpdaterError(f"installed release {version} is incomplete")
    _activate_release_dir(
        release_dir,
        target_install_root=Path(target_install_root or install_root()).expanduser(),
        target_bin_dir=Path(target_bin_dir or (Path.home() / ".local" / "bin")).expanduser(),
    )


def _activate_release_dir(
    release_dir: Path,
    *,
    target_install_root: Path,
    target_bin_dir: Path,
) -> None:
    _point_current_link(release_dir, target_install_root=target_install_root)
    _write_wrapper(target_bin_dir / "multishell", target_install_root=target_install_root)


def _point_current_link(release_dir: Path, *, target_install_root: Path) -> None:
    current_link = _current_link(target_install_root)
    temporary_link = target_install_root / f".current-{uuid.uuid4().hex[:8]}"
    try:
        temporary_link.symlink_to(Path("releases") / release_dir.name, target_is_directory=True)
        os.replace(temporary_link, current_link)
    finally:
        if temporary_link.exists() or temporary_link.is_symlink():
            temporary_link.unlink()


def _write_wrapper(wrapper_path: Path, *, target_install_root: Path) -> None:
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f'INSTALL_ROOT="${{MULTISHELL_INSTALL_ROOT:-{str(target_install_root)}}}"',
                'exec "$INSTALL_ROOT/current/venv/bin/python" -m multishell "$@"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    wrapper_path.chmod(0o755)


def _release_dir(version: str, target_install_root: Path | None = None) -> Path:
    return _releases_dir(target_install_root) / version


def _releases_dir(target_install_root: Path | None = None) -> Path:
    return Path(target_install_root or install_root()).expanduser() / "releases"


def _current_link(target_install_root: Path | None = None) -> Path:
    return Path(target_install_root or install_root()).expanduser() / "current"


def _release_dir_ready(release_dir: Path) -> bool:
    return (release_dir / "venv" / "bin" / "python").exists() and (release_dir / "app" / "pyproject.toml").exists()


def _default_rollback_target(versions: list[str], current: str | None) -> str | None:
    if not versions:
        return None
    if current is None:
        return versions[-1]
    ordered = sorted(versions, key=_version_key)
    if current not in ordered:
        return ordered[-1]
    index = ordered.index(current)
    if index == 0:
        return None
    return ordered[index - 1]


def _update_state_path() -> Path:
    return state_root() / "update-state.json"


def _running_from_managed_install(target_install_root: Path | None = None) -> bool:
    selected_install_root = Path(target_install_root or install_root()).expanduser().resolve()
    candidates: list[Path] = []
    executable = str(sys.executable).strip()
    if executable:
        candidates.append(Path(executable).expanduser())
    module_path = Path(__file__).expanduser()
    candidates.append(module_path)

    for candidate in candidates:
        paths_to_check = [candidate]
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = None
        if resolved is not None and resolved != candidate:
            paths_to_check.append(resolved)
        for path in paths_to_check:
            if path == selected_install_root or selected_install_root in path.parents:
                return True
    return False


def _load_update_state() -> dict[str, object]:
    path = _update_state_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _record_update_check(*, now: float | None = None) -> None:
    path = _update_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    state = _load_update_state()
    state["last_checked_at"] = float(time.time() if now is None else now)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _should_check_for_update(*, now: float | None = None) -> bool:
    state = _load_update_state()
    last_checked_at = state.get("last_checked_at")
    if not isinstance(last_checked_at, (int, float)):
        return True
    current_time = float(time.time() if now is None else now)
    return current_time - float(last_checked_at) >= STARTUP_CHECK_INTERVAL_SECONDS


def _auto_update_mode() -> str:
    raw = os.environ.get(ENV_AUTO_UPDATE, AUTO_UPDATE_MODE_CHECK).strip().lower()
    if raw in {"0", "false", "no"}:
        return AUTO_UPDATE_MODE_OFF
    if raw in {"1", "true", "yes"}:
        return AUTO_UPDATE_MODE_APPLY
    if raw in {AUTO_UPDATE_MODE_CHECK, AUTO_UPDATE_MODE_APPLY, AUTO_UPDATE_MODE_OFF}:
        return raw
    return AUTO_UPDATE_MODE_CHECK
