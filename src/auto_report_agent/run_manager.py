from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from auto_report_agent.settings import RUNS_DIR

_RUN_ID_PATTERN = re.compile(r"^[0-9A-Za-z_-]{8,80}$")


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    directory: Path
    report_file: Path
    metadata_file: Path


def new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{secrets.token_hex(6)}"


def create_run_paths(*, run_id: str | None = None, runs_dir: Path = RUNS_DIR) -> RunPaths:
    """Create an isolated directory for one report-generation run."""
    resolved_id = run_id or new_run_id()
    if not _RUN_ID_PATTERN.fullmatch(resolved_id):
        raise ValueError("run_id 只能包含字母、数字、下划线和连字符，长度为 8-80。")

    root = runs_dir.resolve()
    directory = (root / resolved_id).resolve()
    try:
        directory.relative_to(root)
    except ValueError as exc:  # defensive: the regex already rejects separators
        raise ValueError("运行目录必须位于 runs 根目录内。") from exc

    directory.mkdir(parents=True, exist_ok=False)
    return RunPaths(
        run_id=resolved_id,
        directory=directory,
        report_file=directory / "final_report.md",
        metadata_file=directory / "run.json",
    )


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    try:
        temp.write_text(content, encoding="utf-8")
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def write_run_metadata(paths: RunPaths, metadata: dict[str, object]) -> None:
    payload = {"run_id": paths.run_id, **metadata}
    write_text_atomic(
        paths.metadata_file,
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
