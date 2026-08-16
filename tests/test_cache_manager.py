import os
import time
from pathlib import Path

import pytest

from auto_report_agent import cache_manager
from auto_report_agent.cache_manager import (
    _ensure_inside_project,
    _last_touched,
    _pycache_dirs,
    enforce_retention,
    format_bytes,
    scan_cache,
    total_cache_bytes,
)
from auto_report_agent.settings import PROJECT_ROOT

DAY = 86400


def test_format_bytes_units():
    assert format_bytes(0) == "0 B"
    assert format_bytes(512) == "512 B"
    assert format_bytes(2048).endswith("KB")
    assert format_bytes(5 * 1024 * 1024).endswith("MB")


def test_ensure_inside_project_allows_subpath():
    _ensure_inside_project(PROJECT_ROOT / "output")


def test_ensure_inside_project_rejects_outside(tmp_path: Path):
    outside = tmp_path / "evil"
    outside.mkdir()
    with pytest.raises(ValueError, match="拒绝清理"):
        _ensure_inside_project(outside)


def test_ensure_inside_project_rejects_traversal():
    with pytest.raises(ValueError):
        _ensure_inside_project(PROJECT_ROOT / ".." / ".." / "etc")


def test_scan_cache_returns_all_targets():
    stats = scan_cache()
    keys = {item.key for item in stats}
    assert {
        "literature_cache",
        "crewai_storage",
        "generated_outputs",
        "pycache",
    }.issubset(keys)


def test_total_cache_bytes_is_non_negative():
    assert total_cache_bytes() >= 0


# --- pruned __pycache__ discovery -------------------------------------------


def _make_pycache(root: Path, *parts: str) -> Path:
    target = root.joinpath(*parts, "__pycache__")
    target.mkdir(parents=True)
    (target / "module.cpython-311.pyc").write_bytes(b"x")
    return target


def test_pycache_scan_skips_dependency_and_vcs_trees(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cache_manager, "PROJECT_ROOT", tmp_path)
    wanted = _make_pycache(tmp_path, "src", "auto_report_agent")
    for ignored in (".venv", ".git", "node_modules", ".pip-cache"):
        _make_pycache(tmp_path, ignored, "somepackage")

    found = _pycache_dirs()

    assert found == [wanted]


def test_pycache_scan_does_not_descend_into_pycache(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cache_manager, "PROJECT_ROOT", tmp_path)
    outer = _make_pycache(tmp_path, "pkg")
    (outer / "__pycache__").mkdir()

    assert _pycache_dirs() == [outer]


# --- retention ---------------------------------------------------------------


@pytest.fixture
def retention_roots(monkeypatch, tmp_path: Path):
    runs = tmp_path / "output" / "runs"
    namespaces = tmp_path / "output" / "literature_cache" / "namespaces"
    runs.mkdir(parents=True)
    namespaces.mkdir(parents=True)
    monkeypatch.setattr(cache_manager, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cache_manager, "retention_roots", lambda: [runs, namespaces])
    return runs, namespaces


def _aged_dir(parent: Path, name: str, *, age_days: float, now: float) -> Path:
    directory = parent / name
    directory.mkdir()
    payload = directory / "final_report.md"
    payload.write_text("x" * 100, encoding="utf-8")
    stamp = now - age_days * DAY
    for path in (payload, directory):
        os.utime(path, (stamp, stamp))
    return directory


def test_expired_directories_are_removed(retention_roots):
    runs, namespaces = retention_roots
    now = time.time()
    stale_run = _aged_dir(runs, "old_run", age_days=30, now=now)
    fresh_run = _aged_dir(runs, "new_run", age_days=1, now=now)
    stale_ns = _aged_dir(namespaces, "old_ns", age_days=8, now=now)

    result = enforce_retention(retention_days=7, now=now)

    assert not stale_run.exists()
    assert not stale_ns.exists()
    assert fresh_run.exists(), "entries inside the window must be kept"
    assert result.removed_dirs == 2
    assert result.removed_bytes == 200
    assert result.errors == []


def test_retention_is_disabled_by_zero(retention_roots):
    runs, _ = retention_roots
    now = time.time()
    ancient = _aged_dir(runs, "ancient", age_days=400, now=now)

    result = enforce_retention(retention_days=0, now=now)

    assert ancient.exists()
    assert result.removed_dirs == 0


def test_retention_window_comes_from_env(monkeypatch, retention_roots):
    runs, _ = retention_roots
    now = time.time()
    directory = _aged_dir(runs, "three_days_old", age_days=3, now=now)

    monkeypatch.setenv("OUTPUT_RETENTION_DAYS", "7")
    assert enforce_retention(now=now).removed_dirs == 0
    assert directory.exists()

    monkeypatch.setenv("OUTPUT_RETENTION_DAYS", "1")
    assert enforce_retention(now=now).removed_dirs == 1
    assert not directory.exists()


def test_recent_nested_write_keeps_the_directory(retention_roots):
    """A stale top directory with fresh contents is still in use."""
    _, namespaces = retention_roots
    now = time.time()
    namespace = _aged_dir(namespaces, "session", age_days=30, now=now)
    active = namespace / "chunks"
    active.mkdir()

    assert _last_touched(namespace) >= now - DAY
    assert enforce_retention(retention_days=7, now=now).removed_dirs == 0
    assert namespace.exists()


def test_retention_ignores_loose_files(retention_roots):
    runs, _ = retention_roots
    now = time.time()
    loose = runs / "stray.md"
    loose.write_text("x", encoding="utf-8")
    os.utime(loose, (now - 90 * DAY, now - 90 * DAY))

    result = enforce_retention(retention_days=7, now=now)

    assert loose.exists(), "only whole run/namespace directories expire"
    assert result.removed_dirs == 0


def test_retention_tolerates_a_missing_root(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cache_manager, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cache_manager, "retention_roots", lambda: [tmp_path / "nope"])

    assert enforce_retention(retention_days=1).removed_dirs == 0
