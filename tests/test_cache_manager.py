from pathlib import Path

import pytest

from auto_report_agent.cache_manager import (
    _ensure_inside_project,
    format_bytes,
    scan_cache,
    total_cache_bytes,
)
from auto_report_agent.settings import PROJECT_ROOT


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
