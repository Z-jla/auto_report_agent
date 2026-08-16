from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from auto_report_agent.settings import (
    CREWAI_STORAGE_DIR,
    LITERATURE_CACHE_DIR,
    LOCALAPPDATA_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RUNS_DIR,
    env_int,
)

DEFAULT_RETENTION_DAYS = 7

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


def _directory_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def _last_touched(path: Path) -> float:
    """Best-effort "recently used" time for a directory.

    Writing a nested file only bumps the mtime of its immediate parent, so the
    top directory alone can look stale while the tree is in active use. Looking
    one level down is cheap and matches how runs and cache namespaces are laid
    out.
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
    return newest


def retention_roots() -> list[Path]:
    """Directories holding one expirable child per run or per browser session."""
    return [RUNS_DIR, LITERATURE_CACHE_DIR / "namespaces"]


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
    if days <= 0:
        return RetentionResult(0, 0, [])

    cutoff = (now if now is not None else time.time()) - days * 86400
    removed_dirs = 0
    removed_bytes = 0
    errors: list[str] = []

    for root in retention_roots():
        if not root.is_dir():
            continue
        try:
            children = sorted(root.iterdir())
        except OSError as exc:
            errors.append(f"{root}: {type(exc).__name__}: {exc}")
            continue

        for child in children:
            if not child.is_dir() or _last_touched(child) >= cutoff:
                continue
            try:
                _ensure_inside_project(child)
                size = _directory_bytes(child)
                shutil.rmtree(child)
            except Exception as exc:  # noqa: BLE001 - one stuck entry must not break startup
                errors.append(f"{child.name}: {type(exc).__name__}: {exc}")
                continue
            removed_dirs += 1
            removed_bytes += size

    return RetentionResult(removed_dirs, removed_bytes, errors)


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
