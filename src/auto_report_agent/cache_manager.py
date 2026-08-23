from __future__ import annotations

import os
import shutil
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from auto_report_agent.settings import (
    CREWAI_STORAGE_DIR,
    LITERATURE_CACHE_DIR,
    LOCALAPPDATA_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RUNS_DIR,
    crewai_task_output_persistence_enabled,
    env_int,
)

DEFAULT_RETENTION_DAYS = 7
DEFAULT_RETENTION_SWEEP_INTERVAL_SECONDS = 60 * 60
RETENTION_HEARTBEAT_NAME = ".heartbeat"
CREWAI_TASK_OUTPUT_DB_NAME = "latest_kickoff_task_outputs.db"
_RETENTION_QUARANTINE_PREFIX = ".retention-expired-"

_RETENTION_SWEEP_LOCK = threading.Lock()
_RETENTION_ENTRY_LOCK = threading.RLock()
_last_retention_sweep: float | None = None

# Directories that are large and never worth walking from the app. The README
# tells users to create .venv inside the repository and .git grows without
# bound; any __pycache__ inside them belongs to dependencies, not this project.
_UNWALKED_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        ".pip-cache",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
        "site-packages",
    }
)


@dataclass(frozen=True)
class CacheTarget:
    key: str
    label: str
    path: Path
    description: str
    default_selected: bool = False


@dataclass(frozen=True)
class CacheStats:
    key: str
    label: str
    path: Path
    description: str
    exists: bool
    files: int
    dirs: int
    bytes: int
    default_selected: bool = False


@dataclass(frozen=True)
class CacheCleanResult:
    key: str
    label: str
    deleted_bytes: int
    deleted_files: int
    deleted_dirs: int
    error: str = ""


def cache_targets() -> list[CacheTarget]:
    """Return cache/output locations that are safe to manage from the app."""
    return [
        CacheTarget(
            key="literature_cache",
            label="文献分阶段缓存",
            path=LITERATURE_CACHE_DIR,
            description="保存文献分块摘要、单篇合并摘要和多篇综合摘要。可安全清理，之后同一文献会重新调用模型分析。",
            default_selected=True,
        ),
        CacheTarget(
            key="crewai_storage",
            label="CrewAI 本地缓存",
            path=CREWAI_STORAGE_DIR,
            description="CrewAI 本地状态/缓存目录。清理后可能需要重新初始化 CrewAI 状态。",
        ),
        CacheTarget(
            key="localappdata",
            label="项目 LOCALAPPDATA",
            path=LOCALAPPDATA_DIR,
            description="项目内替代 Windows LOCALAPPDATA 的目录，通常可清理。",
        ),
        CacheTarget(
            key="pip_cache",
            label="项目 pip 缓存",
            path=PROJECT_ROOT / ".pip-cache",
            description="项目内 pip 下载缓存。清理不影响运行，只影响后续安装速度。",
        ),
        CacheTarget(
            key="tmp",
            label="项目临时目录",
            path=PROJECT_ROOT / ".tmp",
            description="临时文件目录。通常可安全清理。",
            default_selected=True,
        ),
        CacheTarget(
            key="generated_runs",
            label="独立运行报告",
            path=RUNS_DIR,
            description="每次生成任务的独立报告与元数据。清理后无法从服务器恢复。",
        ),
        CacheTarget(
            key="generated_outputs",
            label="生成输出文件",
            path=OUTPUT_DIR,
            description="清理 output 下的 md/pdf/docx/json 文件，不删除 literature_cache 目录。",
        ),
        CacheTarget(
            key="pycache",
            label="Python __pycache__",
            path=PROJECT_ROOT,
            description="递归清理项目内所有 __pycache__ 目录。",
        ),
    ]


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GB"


def scan_cache_target(target: CacheTarget) -> CacheStats:
    files = 0
    dirs = 0
    total = 0
    path = target.path

    if target.key == "generated_outputs":
        for item in _generated_output_files():
            files += 1
            try:
                total += item.stat().st_size
            except OSError:
                pass
        return CacheStats(
            key=target.key,
            label=target.label,
            path=path,
            description=target.description,
            exists=path.exists(),
            files=files,
            dirs=0,
            bytes=total,
            default_selected=target.default_selected,
        )

    if target.key == "pycache":
        for cache_dir in _pycache_dirs():
            dirs += 1
            for item in cache_dir.rglob("*"):
                if item.is_file():
                    files += 1
                    try:
                        total += item.stat().st_size
                    except OSError:
                        pass
                elif item.is_dir():
                    dirs += 1
        return CacheStats(
            key=target.key,
            label=target.label,
            path=path,
            description=target.description,
            exists=path.exists(),
            files=files,
            dirs=dirs,
            bytes=total,
            default_selected=target.default_selected,
        )

    if not path.exists():
        return CacheStats(
            key=target.key,
            label=target.label,
            path=path,
            description=target.description,
            exists=False,
            files=0,
            dirs=0,
            bytes=0,
            default_selected=target.default_selected,
        )

    if path.is_file():
        try:
            total = path.stat().st_size
        except OSError:
            total = 0
        return CacheStats(
            key=target.key,
            label=target.label,
            path=path,
            description=target.description,
            exists=True,
            files=1,
            dirs=0,
            bytes=total,
            default_selected=target.default_selected,
        )

    for item in path.rglob("*"):
        if item.is_file():
            files += 1
            try:
                total += item.stat().st_size
            except OSError:
                pass
        elif item.is_dir():
            dirs += 1

    return CacheStats(
        key=target.key,
        label=target.label,
        path=path,
        description=target.description,
        exists=True,
        files=files,
        dirs=dirs,
        bytes=total,
        default_selected=target.default_selected,
    )


def scan_cache() -> list[CacheStats]:
    return [scan_cache_target(target) for target in cache_targets()]


def clean_cache(keys: set[str]) -> list[CacheCleanResult]:
    targets = {target.key: target for target in cache_targets()}
    results: list[CacheCleanResult] = []

    for key in keys:
        target = targets.get(key)
        if target is None:
            results.append(
                CacheCleanResult(
                    key=key,
                    label=key,
                    deleted_bytes=0,
                    deleted_files=0,
                    deleted_dirs=0,
                    error="未知缓存类别",
                )
            )
            continue

        before = scan_cache_target(target)
        try:
            if key == "generated_outputs":
                _delete_files(_generated_output_files())
            elif key == "pycache":
                _delete_paths(_pycache_dirs())
            elif target.path.exists():
                _delete_paths([target.path])
        except Exception as exc:  # noqa: BLE001 - user-facing cleanup tool should report all failures
            results.append(
                CacheCleanResult(
                    key=key,
                    label=target.label,
                    deleted_bytes=0,
                    deleted_files=0,
                    deleted_dirs=0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        results.append(
            CacheCleanResult(
                key=key,
                label=target.label,
                deleted_bytes=before.bytes,
                deleted_files=before.files,
                deleted_dirs=before.dirs,
            )
        )

    return results


def total_cache_bytes(stats: list[CacheStats] | None = None) -> int:
    stats = stats if stats is not None else scan_cache()
    return sum(item.bytes for item in stats)


@dataclass(frozen=True)
class RetentionResult:
    removed_dirs: int
    removed_bytes: int
    errors: list[str]
    removed_files: int = 0
    removed_records: int = 0


def _directory_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def touch_retention_entry(
    path: Path,
    *,
    when: float | None = None,
    allowed_root: Path | None = None,
) -> Path:
    """Refresh an expirable run/namespace without walking its content tree.

    Writers call this at the root of the entry they are actively using.  The
    marker makes a periodic sweep O(number of entries) in the common case and is
    safe to touch concurrently.  ``when`` exists for deterministic tests and
    maintenance/migration tools.
    """
    with _RETENTION_ENTRY_LOCK:
        if allowed_root is None:
            _ensure_inside_project(path)
        else:
            resolved_path = path.resolve()
            resolved_root = allowed_root.resolve()
            try:
                resolved_path.relative_to(resolved_root)
            except ValueError as exc:
                raise ValueError(f"拒绝触碰允许根目录之外的路径：{path}") from exc
        path.mkdir(parents=True, exist_ok=True)
        marker = path / RETENTION_HEARTBEAT_NAME
        marker.touch(exist_ok=True)
        if when is not None:
            os.utime(marker, (when, when))
        return marker


def _last_touched(path: Path, *, cutoff: float | None = None) -> float:
    """Best-effort "recently used" time for a directory.

    A heartbeat (or another fresh direct child) is the fast path.  Entries that
    still look expired are recursively checked before deletion so caches made
    by older versions -- and writes that happened just before heartbeat support
    was added -- are not mistaken for stale data.  Thus a normal sweep avoids a
    deep scan of active namespaces while retaining a correctness fallback.
    """
    try:
        newest = path.stat().st_mtime
    except OSError:
        return 0.0
    try:
        for child in path.iterdir():
            try:
                newest = max(newest, child.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        pass

    if cutoff is not None and newest >= cutoff:
        return newest

    # A real literature cache is namespace/document/chunks/file.  Only checking
    # direct children misses a fresh chunk because neither its namespace nor its
    # document directory gets an mtime update.  Walk only entries that did not
    # already prove themselves fresh via their heartbeat/direct metadata.
    for root, dirnames, filenames in os.walk(path):
        for name in (*dirnames, *filenames):
            try:
                newest = max(newest, (Path(root) / name).stat().st_mtime)
            except OSError:
                continue
        if cutoff is not None and newest >= cutoff:
            break
    return newest


def retention_roots() -> list[Path]:
    """Directories holding one expirable child per run or per browser session."""
    return [RUNS_DIR, LITERATURE_CACHE_DIR / "namespaces"]


def _is_retention_quarantine(path: Path) -> bool:
    """Return whether an entry was already detached by an earlier sweep.

    The second clause recognises names produced by the first quarantine
    implementation. Reusing an existing quarantine is important: repeatedly
    appending an ``.expired-*`` suffix after a locked-file deletion failure can
    eventually exceed the platform's filename-component limit.
    """
    name = path.name
    return name.startswith(_RETENTION_QUARANTINE_PREFIX) or (
        name.startswith(".") and ".expired-" in name
    )


def enforce_retention(
    *, retention_days: int | None = None, now: float | None = None
) -> RetentionResult:
    """Delete run and literature-cache directories older than the retention window.

    Generated runs and per-session literature caches otherwise grow without
    bound, and public deployments hide the manual cleanup UI entirely, so this is
    their only reclaim path. Evicting an entry that is still referenced is safe:
    the next request recomputes it.

    Set ``OUTPUT_RETENTION_DAYS=0`` to disable the sweep.
    """
    days = (
        retention_days
        if retention_days is not None
        else env_int("OUTPUT_RETENTION_DAYS", DEFAULT_RETENTION_DAYS, 0, 365)
    )
    current_time = now if now is not None else time.time()
    cutoff = current_time - days * 86400 if days > 0 else None
    removed_dirs = 0
    removed_bytes = 0
    removed_files = 0
    removed_records = 0
    errors: list[str] = []

    # Disabling CrewAI persistence is a privacy policy, not a retention-window
    # preference: scrub a database left by an earlier/local deployment even if
    # OUTPUT_RETENTION_DAYS=0.  When replay is enabled, rows follow the same
    # retention window as generated runs and literature caches.
    storage_result = _enforce_crewai_task_output_retention(cutoff=cutoff)
    removed_bytes += storage_result[0]
    removed_files += storage_result[1]
    removed_records += storage_result[2]
    errors.extend(storage_result[3])

    if days <= 0:
        return RetentionResult(
            removed_dirs,
            removed_bytes,
            errors,
            removed_files,
            removed_records,
        )

    assert cutoff is not None  # narrowed from the days > 0 branch above
    for root in retention_roots():
        if not root.is_dir():
            continue
        try:
            children = sorted(root.iterdir())
        except OSError as exc:
            errors.append(f"{root}: {type(exc).__name__}: {exc}")
            continue

        for child in children:
            # Serialize the final freshness check with in-process heartbeat
            # writers. If a stale namespace becomes active while a sweep is
            # deciding, either deletion completes before its first heartbeat
            # or the fresh heartbeat wins and the entry is retained.
            with _RETENTION_ENTRY_LOCK:
                if not child.is_dir():
                    continue
                already_quarantined = _is_retention_quarantine(child)
                if not already_quarantined and _last_touched(child, cutoff=cutoff) >= cutoff:
                    continue
                try:
                    _ensure_inside_project(child)
                    if already_quarantined:
                        quarantine = child
                    else:
                        # Keep the detached name independent of the source name
                        # so retries remain bounded even after repeated failures.
                        quarantine = child.with_name(
                            f"{_RETENTION_QUARANTINE_PREFIX}{os.getpid()}-{time.time_ns()}"
                        )
                        child.replace(quarantine)
                except Exception as exc:  # noqa: BLE001 - keep sweeping other entries
                    errors.append(f"{child.name}: {type(exc).__name__}: {exc}")
                    continue

            # The atomic rename makes the stale entry unreachable to new
            # writers. Potentially slow sizing/removal stays outside the
            # heartbeat lock, so other sessions can keep touching their own
            # namespaces while a large expired tree is reclaimed.
            try:
                size = _directory_bytes(quarantine)
                shutil.rmtree(quarantine)
            except Exception as exc:  # noqa: BLE001 - keep sweeping other entries
                errors.append(f"{quarantine.name}: {type(exc).__name__}: {exc}")
                continue
            removed_dirs += 1
            removed_bytes += size

    return RetentionResult(
        removed_dirs,
        removed_bytes,
        errors,
        removed_files,
        removed_records,
    )


def enforce_retention_periodically(
    *,
    min_interval_seconds: int | None = None,
    retention_days: int | None = None,
    now: float | None = None,
    monotonic_now: float | None = None,
) -> RetentionResult | None:
    """Run retention at a process-wide, thread-safe, rate-limited cadence.

    Streamlit reruns and CLI commands can both call this on every entry.  At
    most one thread performs a sweep, and later calls continue to run after the
    interval instead of relying on a browser-session boolean.  ``None`` means a
    sweep was already recent or is currently in progress.
    """
    interval = (
        min_interval_seconds
        if min_interval_seconds is not None
        else env_int(
            "OUTPUT_RETENTION_SWEEP_INTERVAL_SECONDS",
            DEFAULT_RETENTION_SWEEP_INTERVAL_SECONDS,
            1,
            7 * 86400,
        )
    )
    interval = max(0, interval)
    tick = time.monotonic() if monotonic_now is None else monotonic_now

    if not _RETENTION_SWEEP_LOCK.acquire(blocking=False):
        return None
    try:
        global _last_retention_sweep
        if _last_retention_sweep is not None and tick - _last_retention_sweep < interval:
            return None
        result = enforce_retention(retention_days=retention_days, now=now)
        _last_retention_sweep = tick
        return result
    finally:
        _RETENTION_SWEEP_LOCK.release()


def _enforce_crewai_task_output_retention(
    *, cutoff: float | None
) -> tuple[int, int, int, list[str]]:
    """Scrub CrewAI's shared task-output SQLite storage without reading payloads.

    Returns ``(removed_bytes, removed_files, removed_records, errors)``.  Public
    deployments use a no-op task handler and remove any database artifacts left
    by an earlier version.  Explicitly persistent/local deployments retain the
    database but securely delete records older than ``cutoff``.
    """
    database = CREWAI_STORAGE_DIR / CREWAI_TASK_OUTPUT_DB_NAME
    sidecars = [
        Path(f"{database}-journal"),
        Path(f"{database}-wal"),
        Path(f"{database}-shm"),
    ]

    if not crewai_task_output_persistence_enabled():
        removed_bytes = 0
        removed_files = 0
        errors: list[str] = []
        for path in (database, *sidecars):
            if not path.exists():
                continue
            try:
                _ensure_inside_project(path)
                try:
                    removed_bytes += path.stat().st_size
                except OSError:
                    pass
                path.unlink()
                removed_files += 1
            except Exception as exc:  # noqa: BLE001 - report and continue with sidecars
                errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
        return removed_bytes, removed_files, 0, errors

    if cutoff is None or not database.is_file():
        return 0, 0, 0, []

    try:
        _ensure_inside_project(database)
        cutoff_utc = datetime.fromtimestamp(cutoff, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        # sqlite3.Connection.__exit__ commits/rolls back but does not close the
        # file handle. closing() matters on Windows, where an unclosed handle
        # prevents later retention/deployment operations from moving the DB.
        with closing(sqlite3.connect(database, timeout=2)) as connection:
            table_exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='latest_kickoff_task_outputs'"
            ).fetchone()
            if table_exists is None:
                return 0, 0, 0, []
            # The query touches timestamps only; no user input or output content
            # is fetched into this process. secure_delete handles rows removed
            # here, while VACUUM below also rebuilds freelist pages left by
            # CrewAI's own connections before this retention sweep.
            connection.execute("PRAGMA secure_delete=ON")
            cursor = connection.execute(
                "DELETE FROM latest_kickoff_task_outputs WHERE timestamp < ?",
                (cutoff_utc,),
            )
            removed_records = max(0, cursor.rowcount)
            connection.commit()
            maintenance_errors: list[str] = []
            try:
                connection.execute("VACUUM")
            except sqlite3.Error as exc:
                maintenance_errors.append(f"VACUUM {type(exc).__name__}: {exc}")
            try:
                checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if checkpoint is not None and checkpoint[0]:
                    maintenance_errors.append("WAL checkpoint remained busy")
            except sqlite3.Error as exc:
                maintenance_errors.append(f"WAL checkpoint {type(exc).__name__}: {exc}")
        return 0, 0, removed_records, maintenance_errors
    except Exception as exc:  # noqa: BLE001 - a busy/corrupt DB must not break startup
        return 0, 0, 0, [f"{database.name}: {type(exc).__name__}: {exc}"]


def _generated_output_files() -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    allowed_suffixes = {".md", ".pdf", ".docx", ".json"}
    files: list[Path] = []
    for item in OUTPUT_DIR.iterdir():
        if item.is_file() and item.suffix.lower() in allowed_suffixes:
            files.append(item)
    return files


def _pycache_dirs() -> list[Path]:
    """Locate this project's __pycache__ directories without walking dependencies.

    ``rglob`` would descend into .git and a repository-local .venv, which is tens
    of thousands of entries; this runs on every Streamlit rerun, so the walk is
    pruned to the project's own tree.
    """
    found: list[Path] = []
    for root, dirnames, _filenames in os.walk(PROJECT_ROOT):
        found.extend(Path(root) / name for name in dirnames if name == "__pycache__")
        # os.walk honours in-place edits to dirnames, so this prunes the descent.
        dirnames[:] = [
            name for name in dirnames if name != "__pycache__" and name not in _UNWALKED_DIR_NAMES
        ]
    return found


def _delete_files(paths: list[Path]) -> None:
    for path in paths:
        _ensure_inside_project(path)
        if path.exists() and path.is_file():
            path.unlink()


def _delete_paths(paths: list[Path]) -> None:
    for path in paths:
        _ensure_inside_project(path)
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _ensure_inside_project(path: Path) -> None:
    root = PROJECT_ROOT.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"拒绝清理项目目录之外的路径：{resolved}") from exc
