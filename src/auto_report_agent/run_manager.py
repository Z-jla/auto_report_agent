from __future__ import annotations

import json
import logging
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from auto_report_agent.settings import RUNS_DIR

LOGGER = logging.getLogger(__name__)

_RUN_ID_PATTERN = re.compile(r"^[0-9A-Za-z_-]{8,80}$")


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    directory: Path
    report_file: Path
    metadata_file: Path

    @property
    def draft_file(self) -> Path:
        """Intermediate writer output kept separate from the reviewed report."""
        return self.directory / "draft_report.md"


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


def handle_failed_report_files(paths: RunPaths, *, keep_draft: bool) -> Path | None:
    """Apply the configured failure policy without masking the original error.

    Writers save to ``draft_report.md`` and reviewers save to
    ``final_report.md``. A non-empty final report may already be complete when a
    later persistence hook fails, so it always wins and both artifacts are left
    untouched for recovery. Only when the final report is missing or blank is a
    complete writer draft atomically promoted over it. Returning a path means
    readable output remains available; ``None`` means there was nothing usable
    or preservation was disabled.
    """
    if not keep_draft:
        for path in (paths.draft_file, paths.report_file):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                LOGGER.warning("Could not remove failed report artifact %s: %s", path, exc)
        return None

    report_unreadable = False
    try:
        existing = paths.report_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = ""
    except (OSError, UnicodeError) as exc:
        # An existing final that cannot currently be inspected may still be a
        # complete reviewer output (for example, while antivirus holds it on
        # Windows). Never overwrite an artifact whose contents are unknown.
        LOGGER.warning("Could not inspect failed final report %s: %s", paths.report_file, exc)
        existing = ""
        report_unreadable = True
    if existing.strip():
        return paths.report_file

    try:
        draft = paths.draft_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        draft = ""
    if draft.strip():
        if report_unreadable:
            return paths.draft_file
        try:
            write_text_atomic(paths.report_file, draft)
            paths.draft_file.unlink(missing_ok=True)
            return paths.report_file
        except OSError as exc:
            # The original task exception is more useful than a secondary file
            # error. Leave the separate draft in place for manual recovery.
            LOGGER.warning("Could not promote failed report draft %s: %s", paths.draft_file, exc)
            return paths.draft_file if paths.draft_file.is_file() else None

    return None


def discard_intermediate_draft(paths: RunPaths) -> None:
    """Best-effort removal after a reviewed final report is safely available."""
    try:
        paths.draft_file.unlink(missing_ok=True)
    except OSError as exc:
        LOGGER.warning("Could not remove intermediate report draft %s: %s", paths.draft_file, exc)


def write_run_metadata(paths: RunPaths, metadata: dict[str, object]) -> None:
    payload = {"run_id": paths.run_id, **metadata}
    write_text_atomic(
        paths.metadata_file,
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
