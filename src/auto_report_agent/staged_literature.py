from __future__ import annotations

import hashlib
import json
import queue
import re
import secrets
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError, Future, as_completed
from dataclasses import dataclass
from pathlib import Path

from auto_report_agent.api_config import (
    ResolvedLLMConfig,
    resolve_llm_env,
    validate_base_url,
)
from auto_report_agent.cache_manager import touch_retention_entry
from auto_report_agent.document_ingest import ParsedDocument
from auto_report_agent.openai_web_search_tool import OpenAIWebSearchTool
from auto_report_agent.settings import (
    LITERATURE_CACHE_DIR,
    env_bool,
    env_int,
    initialize_runtime,
)

ProgressCallback = Callable[[str], None]
CACHE_VERSION = "staged-literature-v5"
PROMPT_VERSION = "literature-prompts-2026-08-23-v3"
initialize_runtime()
CACHE_DIR = LITERATURE_CACHE_DIR

# Environment settings below remain configurable, but these constants are
# deliberately non-configurable ceilings. They prevent an accidental setting or
# an unexpectedly verbose OpenAI-compatible endpoint from rebuilding the former
# worst case (hundreds of thousands of characters passed to the final Crew).
HARD_MAX_METADATA_CHARS = 4_000
HARD_MAX_CHUNK_PROMPT_CHARS = 24_000
HARD_MAX_CHUNK_SUMMARY_CHARS = 12_000
HARD_MAX_DOC_PROMPT_CHARS = 60_000
HARD_MAX_DOC_SUMMARY_CHARS = 16_000
HARD_MAX_CORPUS_PROMPT_CHARS = 80_000
HARD_MAX_CORPUS_SUMMARY_CHARS = 24_000
HARD_MAX_FALLBACK_CHARS = 80_000
HARD_MAX_PAPER_CONTEXT_CHARS = 100_000
HARD_MAX_MODEL_RESPONSE_BYTES = 4 * 1024 * 1024
HARD_MAX_MODEL_ERROR_BYTES = 64 * 1024
_TRUNCATION_NOTICE = "\n\n> [已按上下文预算裁剪；保留开头与结尾]"


@dataclass
class StagedLiteratureResult:
    paper_context: str
    chunk_count: int
    doc_count: int


@dataclass(frozen=True)
class _StageBudgets:
    """Character budgets for every untrusted/model-generated text boundary."""

    metadata_chars: int
    chunk_prompt_chars: int
    chunk_summary_chars: int
    doc_prompt_chars: int
    doc_summary_chars: int
    corpus_prompt_chars: int
    corpus_summary_chars: int
    fallback_chars: int
    paper_context_chars: int


def _hash_text(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:length]


def _credential_fingerprint(api_key: str) -> str:
    """Return a stable, one-way cache partition without persisting the secret."""
    return hashlib.sha256(
        f"staged-literature-credential-v1\0{api_key}".encode("utf-8", errors="replace")
    ).hexdigest()


def _truncate_text(text: str, max_chars: int) -> str:
    """Bound text while retaining both context-setting and concluding content."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= len(_TRUNCATION_NOTICE) + 8:
        return text[:max_chars]

    available = max_chars - len(_TRUNCATION_NOTICE)
    head_chars = (available * 2) // 3
    tail_chars = available - head_chars
    return f"{text[:head_chars].rstrip()}{_TRUNCATION_NOTICE}{text[-tail_chars:].lstrip()}"[
        :max_chars
    ]


def _allocate_fair_budgets(lengths: Sequence[int], total: int) -> list[int]:
    """Water-fill a total budget so short sections donate space to long ones."""
    allocations = [0] * len(lengths)
    active = set(range(len(lengths)))
    remaining = max(0, total)
    while active and remaining:
        share = remaining // len(active)
        if share == 0:
            for index in sorted(active)[:remaining]:
                allocations[index] += 1
            break
        completed = [index for index in active if lengths[index] <= share]
        if not completed:
            for index in active:
                allocations[index] = share
            for index in sorted(active)[: remaining - share * len(active)]:
                allocations[index] += 1
            break
        for index in completed:
            allocations[index] = lengths[index]
            remaining -= lengths[index]
            active.remove(index)
    return allocations


def _bounded_sections(sections: Sequence[tuple[str, str]], *, max_chars: int) -> str:
    """Format every labelled section within one hard aggregate character cap.

    Each section receives a fair share, rather than a plain prefix truncation
    that would silently discard all middle/end documents. Coverage labels and
    the beginning/end of each summary therefore survive normal budget pressure.
    """
    if not sections or max_chars <= 0:
        return ""

    count = len(sections)
    label_cap = max(8, min(160, max_chars // max(4 * count, 1)))
    labels = [f"## {_truncate_text(label, label_cap)}\n" for label, _ in sections]
    separator_chars = 2 * (count - 1)
    overhead = sum(map(len, labels)) + separator_chars
    if overhead >= max_chars:
        return _truncate_text("\n\n".join(labels), max_chars)

    texts = [text for _, text in sections]
    allocations = _allocate_fair_budgets([len(text) for text in texts], max_chars - overhead)
    rendered = [
        f"{label}{_truncate_text(text, allocation)}"
        for label, text, allocation in zip(labels, texts, allocations, strict=True)
    ]
    return "\n\n".join(rendered)[:max_chars]


def _safe_filename(name: str, max_len: int = 80) -> str:
    stem = Path(name).stem or "document"
    safe = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", stem).strip("._-")
    return (safe or "document")[:max_len]


def _cache_namespace_root(cache_namespace: str) -> Path:
    namespace_hash = _hash_text(cache_namespace or "shared", 16)
    return CACHE_DIR / "namespaces" / namespace_hash


def _cache_root_for_document(document: ParsedDocument, *, cache_namespace: str) -> Path:
    doc_hash = _hash_text(f"{CACHE_VERSION}\n{document.name}\n{document.content}", 16)
    return _cache_namespace_root(cache_namespace) / f"{_safe_filename(document.name)}_{doc_hash}"


def _touch_cache_namespace(cache_namespace: str) -> None:
    """Record recent nested cache activity for namespace-level retention."""
    if env_bool("PAPER_CACHE_ENABLED", True):
        touch_retention_entry(
            _cache_namespace_root(cache_namespace),
            allowed_root=CACHE_DIR,
        )


def _read_cache(path: Path, *, cache_namespace: str) -> str | None:
    if not env_bool("PAPER_CACHE_ENABLED", True):
        return None
    try:
        if path.exists():
            content = path.read_text(encoding="utf-8").strip()
            _touch_cache_namespace(cache_namespace)
            return content or None
    except OSError:
        return None
    return None


def _write_cache(path: Path, content: str, *, cache_namespace: str) -> None:
    if not env_bool("PAPER_CACHE_ENABLED", True):
        return
    _touch_cache_namespace(cache_namespace)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    try:
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(path)
        _touch_cache_namespace(cache_namespace)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _chunk_cache_path(
    *,
    document: ParsedDocument,
    topic: str,
    instruction: str,
    chunk_index: int,
    total_chunks: int,
    chunk_text: str,
    cache_namespace: str,
    cache_identity: dict[str, object],
) -> Path:
    key = _hash_text(
        json.dumps(
            {
                "version": CACHE_VERSION,
                "kind": "chunk",
                "document": document.name,
                "topic_hash": _hash_text(topic, 64),
                "instruction_hash": _hash_text(instruction, 64),
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "chunk_text": chunk_text,
                "runtime": cache_identity,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        20,
    )
    return (
        _cache_root_for_document(document, cache_namespace=cache_namespace)
        / "chunks"
        / f"chunk_{chunk_index:03d}_{key}.md"
    )


def _doc_summary_cache_path(
    *,
    document: ParsedDocument,
    topic: str,
    instruction: str,
    chunk_summaries: Sequence[str],
    cache_namespace: str,
    cache_identity: dict[str, object],
) -> Path:
    key = _hash_text(
        json.dumps(
            {
                "version": CACHE_VERSION,
                "kind": "doc_summary",
                "document": document.name,
                "topic_hash": _hash_text(topic, 64),
                "instruction_hash": _hash_text(instruction, 64),
                "chunk_summaries": list(chunk_summaries),
                "runtime": cache_identity,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        20,
    )
    return (
        _cache_root_for_document(document, cache_namespace=cache_namespace)
        / f"doc_summary_{key}.md"
    )


def _corpus_summary_cache_path(
    *,
    topic: str,
    instruction: str,
    doc_summaries: Sequence[tuple[str, str]],
    cache_namespace: str,
    cache_identity: dict[str, object],
) -> Path:
    key = _hash_text(
        json.dumps(
            {
                "version": CACHE_VERSION,
                "kind": "corpus_summary",
                "topic_hash": _hash_text(topic, 64),
                "instruction_hash": _hash_text(instruction, 64),
                "doc_summaries": list(doc_summaries),
                "runtime": cache_identity,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        20,
    )
    return _cache_namespace_root(cache_namespace) / "corpus" / f"corpus_summary_{key}.md"


def split_text_into_chunks(text: str, chunk_size: int) -> list[str]:
    """按段落尽量自然切块，避免一次请求过长。"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append("\n\n".join(current).strip())
                current = []
                current_len = 0
            for start in range(0, len(paragraph), chunk_size):
                part = paragraph[start : start + chunk_size].strip()
                if part:
                    chunks.append(part)
            continue

        next_len = current_len + len(paragraph) + 2
        if current and next_len > chunk_size:
            chunks.append("\n\n".join(current).strip())
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len = next_len

    if current:
        chunks.append("\n\n".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def select_chunk_coverage(chunks: Sequence[str], max_chunks: int) -> list[tuple[int, str]]:
    """Select evenly distributed chunks and retain their original 1-based positions."""
    if max_chunks < 1:
        raise ValueError("max_chunks 必须至少为 1。")
    if len(chunks) <= max_chunks:
        return list(enumerate(chunks, start=1))
    if max_chunks == 1:
        return [(1, chunks[0])]

    indices = [round(index * (len(chunks) - 1) / (max_chunks - 1)) for index in range(max_chunks)]
    return [(index + 1, chunks[index]) for index in indices]


def chunk_character_ranges(text: str, chunks: Sequence[str]) -> list[tuple[int, int]]:
    """Return best-effort, 1-based inclusive character ranges for sequential chunks."""
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for chunk in chunks:
        start = text.find(chunk, cursor)
        if start < 0:
            start = min(cursor, len(text))
        end = min(len(text), start + len(chunk))
        ranges.append((start + 1, max(start + 1, end)))
        cursor = end
    return ranges


def _runtime_cache_identity(
    *,
    config: ResolvedLLMConfig,
    chunk_size: int,
    max_chunks_per_doc: int,
    max_output_tokens: int,
    merge_output_tokens: int,
    budgets: _StageBudgets | None = None,
) -> dict[str, object]:
    env = config
    identity: dict[str, object] = {
        "cache_version": CACHE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "base_url": env.base_url,
        "model": env.model,
        "api_mode": env.api_mode,
        "api_key_fingerprint": _credential_fingerprint(env.api_key),
        "chunk_size": chunk_size,
        "max_chunks_per_doc": max_chunks_per_doc,
        "max_output_tokens": max_output_tokens,
        "merge_output_tokens": merge_output_tokens,
        "temperature": 0.2,
    }
    if budgets is not None:
        identity["character_budgets"] = {
            "metadata": budgets.metadata_chars,
            "chunk_prompt": budgets.chunk_prompt_chars,
            "chunk_summary": budgets.chunk_summary_chars,
            "doc_prompt": budgets.doc_prompt_chars,
            "doc_summary": budgets.doc_summary_chars,
            "corpus_prompt": budgets.corpus_prompt_chars,
            "corpus_summary": budgets.corpus_summary_chars,
            "fallback": budgets.fallback_chars,
            "paper_context": budgets.paper_context_chars,
        }
    return identity


def _extract_response_text(data: dict) -> str:
    text = OpenAIWebSearchTool._extract_response_text(data)
    if text:
        return text

    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return data["output_text"].strip()

    choices = data.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    return ""


def _call_model(
    prompt: str,
    *,
    config: ResolvedLLMConfig,
    max_output_tokens: int,
    timeout: int,
    retries: int = 1,
    cancelled: threading.Event | None = None,
) -> str:
    env = config

    if not env.api_key:
        raise RuntimeError("缺少 LLM_API_KEY（或 OPENAI_API_KEY），无法进行分阶段文献分析。")
    if not env.base_url:
        raise RuntimeError("缺少 LLM_BASE_URL（或 OPENAI_API_BASE），无法进行分阶段文献分析。")
    if not env.model:
        raise RuntimeError("缺少 LLM_MODEL（或 OPENAI_MODEL_NAME），无法进行分阶段文献分析。")

    try:
        base_url = validate_base_url(env.base_url)
    except ValueError as exc:
        raise RuntimeError(f"Base URL 安全校验未通过：{exc}") from exc

    if env.api_mode == "responses":
        endpoint = f"{base_url}/responses"
        payload = {
            "model": env.model,
            "input": prompt,
            "store": False,
            "max_output_tokens": max_output_tokens,
        }
    else:
        endpoint = f"{base_url}/chat/completions"
        payload = {
            "model": env.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": max_output_tokens,
        }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {env.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "auto-report-agent/0.2 staged-literature",
    }

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        if cancelled is not None and cancelled.is_set():
            raise CancelledError()
        request = urllib.request.Request(
            endpoint,
            data=data,
            method="POST",
            headers=headers,
        )
        try:
            opener = urllib.request.build_opener(_NoRedirectHandler())
            with opener.open(request, timeout=timeout) as response:
                raw_bytes = response.read(HARD_MAX_MODEL_RESPONSE_BYTES + 1)
                if len(raw_bytes) > HARD_MAX_MODEL_RESPONSE_BYTES:
                    raise RuntimeError("模型接口响应超过安全字节上限。")
                raw = raw_bytes.decode("utf-8", errors="replace")
            parsed = json.loads(raw)
            text = _extract_response_text(parsed)
            if text:
                return text
            raise RuntimeError("模型接口返回成功，但没有解析到文本内容。")
        except urllib.error.HTTPError as exc:
            body = exc.read(HARD_MAX_MODEL_ERROR_BYTES + 1).decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {body[:1200]}")
        except CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc

        if attempt < retries:
            delay = min(8, 2 + attempt * 3)
            if cancelled is not None:
                if cancelled.wait(delay):
                    raise CancelledError()
            else:
                time.sleep(delay)

    raise RuntimeError(f"分阶段文献分析调用模型失败：{type(last_error).__name__}: {last_error}")


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _chunk_prompt(
    *,
    topic: str,
    instruction: str,
    document_name: str,
    chunk_index: int,
    total_chunks: int,
    char_start: int,
    char_end: int,
    chunk_text: str,
    metadata_max_chars: int,
    prompt_max_chars: int,
) -> str:
    # Misconfigured metadata must not consume the whole prompt and crowd out the
    # source material. Each free-form field receives at most one eighth.
    field_budget = min(metadata_max_chars, max(100, prompt_max_chars // 8))
    topic = _truncate_text(topic or "未指定", field_budget)
    instruction = _truncate_text(instruction or "无", field_budget)
    document_name = _truncate_text(document_name, 240)
    prefix = f"""
你是严谨的学术文献阅读助手。请只基于下面这一段文献内容做局部摘要，不要编造文中没有的信息。

文献主题/综述方向：{topic}
用户补充要求：{instruction}
文献文件：{document_name}
当前片段：第 {chunk_index}/{total_chunks} 段
提供文本字符范围：{char_start}-{char_end}

请输出 Markdown，包含：
1. 本片段核心内容
2. 涉及的研究问题/背景
3. 方法、模型、数据或实验信息
4. 关键结论或发现
5. 创新点/局限性/待核实信息
6. 可引用的事实依据

如果某项未出现，请写“本片段未明确说明”。

## 文献片段
""".strip()
    available = max(0, prompt_max_chars - len(prefix) - 1)
    return f"{prefix}\n{_truncate_text(chunk_text, available)}"[:prompt_max_chars]


def _available_chunk_source_chars(
    *,
    topic: str,
    instruction: str,
    document_name: str,
    metadata_max_chars: int,
    prompt_max_chars: int,
) -> int:
    """Return a conservative source budget after fixed prompt metadata."""
    empty_prompt = _chunk_prompt(
        topic=topic,
        instruction=instruction,
        document_name=document_name,
        chunk_index=999_999_999,
        total_chunks=999_999_999,
        char_start=999_999_999,
        char_end=999_999_999,
        chunk_text="",
        metadata_max_chars=metadata_max_chars,
        prompt_max_chars=prompt_max_chars,
    )
    return max(1, prompt_max_chars - len(empty_prompt))


def _doc_merge_prompt(
    *,
    topic: str,
    instruction: str,
    document_name: str,
    chunk_summaries: Sequence[str],
    metadata_max_chars: int,
    prompt_max_chars: int,
) -> str:
    field_budget = min(metadata_max_chars, max(100, prompt_max_chars // 8))
    topic = _truncate_text(topic or "未指定", field_budget)
    instruction = _truncate_text(instruction or "无", field_budget)
    document_name = _truncate_text(document_name, 240)
    prefix = f"""
你是学术文献分析专家。下面是同一篇文献的多个片段摘要。请合并为一份“单篇文献结构化摘要”。

文献主题/综述方向：{topic}
用户补充要求：{instruction}
文献文件：{document_name}

请输出 Markdown，必须包含：
- 文献标题/主题（如果摘要中无法判断，请写文件名）
- 研究问题与背景
- 方法/模型/理论框架
- 数据、实验或案例
- 核心结果与结论
- 创新点
- 局限性
- 与用户综述方向的关系
- 可直接用于最终报告的事实依据

要求：只能根据片段摘要合并，不要编造。

## 片段摘要
""".strip()
    available = max(0, prompt_max_chars - len(prefix) - 1)
    sections = [
        (f"片段摘要 {index}", summary) for index, summary in enumerate(chunk_summaries, start=1)
    ]
    return f"{prefix}\n{_bounded_sections(sections, max_chars=available)}"[:prompt_max_chars]


def _corpus_merge_prompt(
    *,
    topic: str,
    instruction: str,
    doc_summaries: Sequence[tuple[str, str]],
    metadata_max_chars: int,
    prompt_max_chars: int,
) -> str:
    field_budget = min(metadata_max_chars, max(100, prompt_max_chars // 8))
    topic = _truncate_text(topic or "未指定", field_budget)
    instruction = _truncate_text(instruction or "无", field_budget)
    prefix = f"""
你是中文文献综述助手。下面是多篇文献的单篇结构化摘要。请生成给后续写作 Agent 使用的“综合文献分析材料”。

文献主题/综述方向：{topic}
用户补充要求：{instruction}

请输出 Markdown，必须包含：
1. 文献总体概览
2. 每篇文献核心摘要
3. 研究问题对比
4. 方法/模型/数据对比
5. 结果与结论对比
6. 创新点与局限性对比
7. 共同趋势、研究空白与后续方向
8. 可直接用于最终报告的事实依据列表

要求：只基于下方摘要，不要添加未出现的信息。

## 单篇文献摘要
""".strip()
    available = max(0, prompt_max_chars - len(prefix) - 1)
    sections = [
        (f"文献 {index}：{name}", summary)
        for index, (name, summary) in enumerate(doc_summaries, start=1)
    ]
    return f"{prefix}\n{_bounded_sections(sections, max_chars=available)}"[:prompt_max_chars]


def _fallback_summary(
    *,
    title: str,
    failure: Exception,
    sections: Sequence[tuple[str, str]],
    max_chars: int,
) -> str:
    prefix = (
        f"# {title}\n\n"
        "> 自动合并步骤失败，以下内容来自已成功缓存的分阶段摘要；"
        f"已应用安全字符预算。失败类型：{type(failure).__name__}\n\n"
    )
    available = max(0, max_chars - len(prefix))
    return f"{prefix}{_bounded_sections(sections, max_chars=available)}"[:max_chars]


def _final_paper_context(
    *,
    coverage_notes: Sequence[tuple[str, str]],
    analysis: str,
    max_chars: int,
) -> str:
    """Always preserve a bounded per-document coverage manifest in final context."""
    prefix = "# 分阶段文献分析结果\n\n## 文献覆盖说明\n"
    analysis_heading = "\n\n## 综合分析\n"
    available = max(0, max_chars - len(prefix) - len(analysis_heading))
    coverage_budget = min(20_000, max(1_500, available // 3))
    coverage = _bounded_sections(coverage_notes, max_chars=coverage_budget)
    analysis_budget = max(0, available - len(coverage))
    bounded_analysis = _truncate_text(analysis, analysis_budget)
    return f"{prefix}{coverage}{analysis_heading}{bounded_analysis}"[:max_chars]


@dataclass(frozen=True)
class _ChunkStageOptions:
    """Settings shared by every chunk-summarization call in one run."""

    topic: str
    instruction: str
    cache_namespace: str
    cache_identity: dict[str, object]
    config: ResolvedLLMConfig
    max_output_tokens: int
    timeout: int
    retries: int
    workers: int
    budgets: _StageBudgets


def _chunk_coverage_note(
    chunk_index: int, total_chunks: int, char_start: int, char_end: int
) -> str:
    return (
        f"> 覆盖片段：原始分块 {chunk_index}/{total_chunks}，提供文本字符 {char_start}-{char_end}"
    )


def _summarize_chunk_uncached(
    *,
    document: ParsedDocument,
    options: _ChunkStageOptions,
    cache_path: Path,
    chunk_index: int,
    total_chunks: int,
    char_start: int,
    char_end: int,
    chunk_text: str,
    cancelled: threading.Event,
) -> str:
    """Call the model for a single chunk and cache the result.

    This runs on a worker thread, so it must not touch the progress callback:
    the Streamlit timeline may only be updated from the thread running the script.
    """
    if cancelled.is_set():
        raise CancelledError()
    summary = _call_model(
        _chunk_prompt(
            topic=options.topic,
            instruction=options.instruction,
            document_name=document.name,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            char_start=char_start,
            char_end=char_end,
            chunk_text=chunk_text,
            metadata_max_chars=options.budgets.metadata_chars,
            prompt_max_chars=options.budgets.chunk_prompt_chars,
        ),
        config=options.config,
        max_output_tokens=options.max_output_tokens,
        timeout=options.timeout,
        retries=options.retries,
        cancelled=cancelled,
    )
    summary = _truncate_text(summary, options.budgets.chunk_summary_chars)
    if cancelled.is_set():
        raise CancelledError()
    _write_cache(cache_path, summary, cache_namespace=options.cache_namespace)
    return summary


def _observe_background_future(future: Future[str]) -> None:
    """Consume late cancellation/errors after a fast-fail executor shutdown."""
    try:
        future.exception()
    except BaseException:  # noqa: BLE001 - callbacks must never leak an exception
        pass


@dataclass(frozen=True)
class _ChunkWorkItem:
    future: Future[str]
    document: ParsedDocument
    options: _ChunkStageOptions
    cache_path: Path
    chunk_index: int
    total_chunks: int
    char_start: int
    char_end: int
    chunk_text: str


def _chunk_worker(
    work_queue: queue.Queue[_ChunkWorkItem | None],
    cancelled: threading.Event,
    failure_lock: threading.Lock,
    first_failure: list[BaseException],
) -> None:
    """Run chunk calls on a daemon worker and contain every late exception.

    ``ThreadPoolExecutor.shutdown(wait=False)`` still registers its non-daemon
    workers for an interpreter-exit join. A slow sibling could therefore make a
    CLI process hang after the caller had already received a fast failure. These
    small, run-scoped daemon workers are intentionally not registered with that
    global exit hook. Every exception is stored in a Future, so none is printed
    as an unhandled background-thread exception.
    """
    while True:
        item = work_queue.get()
        try:
            if item is None:
                return
            if cancelled.is_set() or not item.future.set_running_or_notify_cancel():
                item.future.cancel()
                continue
            try:
                result = _summarize_chunk_uncached(
                    document=item.document,
                    options=item.options,
                    cache_path=item.cache_path,
                    chunk_index=item.chunk_index,
                    total_chunks=item.total_chunks,
                    char_start=item.char_start,
                    char_end=item.char_end,
                    chunk_text=item.chunk_text,
                    cancelled=cancelled,
                )
            except BaseException as exc:  # noqa: BLE001 - contain worker failures
                with failure_lock:
                    if not first_failure:
                        first_failure.append(exc)
                # Do this in the worker before it can dequeue another item.
                # Waiting for the caller to observe the Future leaves a race in
                # which a known-doomed queued API request can still start.
                cancelled.set()
                item.future.set_exception(exc)
            else:
                item.future.set_result(result)
        finally:
            work_queue.task_done()


def _summarize_document_chunks(
    document: ParsedDocument,
    *,
    options: _ChunkStageOptions,
    all_chunks: Sequence[str],
    all_ranges: Sequence[tuple[int, int]],
    selected_chunks: Sequence[tuple[int, str]],
    progress: ProgressCallback | None = None,
) -> list[str]:
    """Summarize one document's selected chunks, running uncached ones in parallel.

    Chunk summaries only depend on their own text, so the ones that need a model
    call are dispatched to a small thread pool instead of being issued one at a
    time. Results are reassembled in document order regardless of completion
    order, and every progress message is emitted from the calling thread.
    """
    total_chunks = len(all_chunks)
    summaries: list[str] = [""] * len(selected_chunks)
    pending: list[tuple[int, int, str, Path]] = []

    # Resolve cache hits up front: they need no network, and settling them here
    # sizes the pool to the work that actually costs a model call.
    for position, (chunk_index, chunk) in enumerate(selected_chunks):
        char_start, char_end = all_ranges[chunk_index - 1]
        cache_path = _chunk_cache_path(
            document=document,
            topic=options.topic,
            instruction=options.instruction,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            chunk_text=chunk,
            cache_namespace=options.cache_namespace,
            cache_identity=options.cache_identity,
        )
        cached_summary = _read_cache(cache_path, cache_namespace=options.cache_namespace)
        if cached_summary:
            cached_summary = _truncate_text(cached_summary, options.budgets.chunk_summary_chars)
            if progress:
                progress(
                    f"命中缓存：《{document.name}》原始片段 {chunk_index}/{total_chunks} "
                    "已完成，跳过模型调用。"
                )
            note = _chunk_coverage_note(chunk_index, total_chunks, char_start, char_end)
            summaries[position] = f"{note}\n\n{cached_summary}"
        else:
            pending.append((position, chunk_index, chunk, cache_path))

    if not pending:
        return summaries

    workers = max(1, min(options.workers, len(pending)))
    if progress:
        progress(
            f"《{document.name}》需要调用模型的片段共 {len(pending)} 个，"
            f"并发 {workers} 个同时分析。"
        )

    futures: dict[Future[str], tuple[int, int, int, int]] = {}
    cancelled = threading.Event()
    failure_lock = threading.Lock()
    first_failure: list[BaseException] = []
    work_queue: queue.Queue[_ChunkWorkItem | None] = queue.Queue()
    worker_threads = [
        threading.Thread(
            target=_chunk_worker,
            args=(work_queue, cancelled, failure_lock, first_failure),
            name=f"chunk-summary-{index + 1}",
            daemon=True,
        )
        for index in range(workers)
    ]
    for thread in worker_threads:
        thread.start()
    try:
        for position, chunk_index, chunk, cache_path in pending:
            char_start, char_end = all_ranges[chunk_index - 1]
            future: Future[str] = Future()
            work_queue.put(
                _ChunkWorkItem(
                    future=future,
                    document=document,
                    options=options,
                    cache_path=cache_path,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    char_start=char_start,
                    char_end=char_end,
                    chunk_text=chunk,
                )
            )
            futures[future] = (position, chunk_index, char_start, char_end)
        for _ in worker_threads:
            work_queue.put(None)

        completed = 0
        for future in as_completed(futures):
            position, chunk_index, char_start, char_end = futures[future]
            summary = future.result()
            completed += 1
            if progress:
                progress(
                    f"已完成《{document.name}》原始片段 {chunk_index}/{total_chunks}"
                    f"（{completed}/{len(pending)}）。"
                )
            note = _chunk_coverage_note(chunk_index, total_chunks, char_start, char_end)
            summaries[position] = f"{note}\n\n{summary}"
    except BaseException as exc:
        # Signal cooperative workers, cancel queued calls, consume late
        # results/errors, and return control without waiting for HTTP siblings.
        cancelled.set()
        for future in futures:
            future.add_done_callback(_observe_background_future)
            future.cancel()
        with failure_lock:
            root_failure = first_failure[0] if first_failure else None
        if isinstance(exc, CancelledError) and root_failure is not None:
            raise root_failure from exc
        raise
    else:
        # Every future completed; joins only let workers consume their sentinels
        # and cannot wait on an outstanding model call at this point.
        for thread in worker_threads:
            thread.join()

    return summaries


def summarize_documents_staged(
    documents: Sequence[ParsedDocument],
    *,
    topic: str,
    instruction: str,
    progress: ProgressCallback | None = None,
    cache_namespace: str = "shared",
    config: ResolvedLLMConfig | None = None,
) -> StagedLiteratureResult:
    """分块阅读文献，生成压缩后的综合文献上下文。

    ``config`` carries one caller's LLM settings; it defaults to the process
    environment, which is what the CLI uses. It also feeds the cache identity, so
    switching model or endpoint does not reuse another configuration's summaries.
    """
    config = config or resolve_llm_env()
    # Claim the namespace before any potentially long model call. A concurrent
    # retention sweep can then never classify this run's cache as stale.
    _touch_cache_namespace(cache_namespace)
    chunk_size = env_int("PAPER_CHUNK_SIZE", 6000, 2000, 12000)
    max_chunks_per_doc = env_int("PAPER_MAX_CHUNKS_PER_DOC", 10, 3, 30)
    max_output_tokens = env_int("PAPER_STAGE_MAX_OUTPUT_TOKENS", 900, 300, 2000)
    merge_output_tokens = env_int("PAPER_STAGE_MERGE_TOKENS", 1400, 500, 3000)
    timeout = env_int("PAPER_STAGE_TIMEOUT", 90, 20, 180)
    retries = env_int("PAPER_STAGE_RETRIES", 1, 0, 3)
    workers = env_int("PAPER_STAGE_CONCURRENCY", 4, 1, 16)
    budgets = _StageBudgets(
        metadata_chars=env_int("PAPER_STAGE_METADATA_CHARS", 1_200, 200, HARD_MAX_METADATA_CHARS),
        chunk_prompt_chars=env_int(
            "PAPER_STAGE_CHUNK_PROMPT_CHARS", 16_000, 4_000, HARD_MAX_CHUNK_PROMPT_CHARS
        ),
        chunk_summary_chars=env_int(
            "PAPER_STAGE_CHUNK_SUMMARY_CHARS",
            min(max_output_tokens * 4, HARD_MAX_CHUNK_SUMMARY_CHARS),
            1_200,
            HARD_MAX_CHUNK_SUMMARY_CHARS,
        ),
        doc_prompt_chars=env_int(
            "PAPER_STAGE_DOC_PROMPT_CHARS", 32_000, 8_000, HARD_MAX_DOC_PROMPT_CHARS
        ),
        doc_summary_chars=env_int(
            "PAPER_STAGE_DOC_SUMMARY_CHARS",
            min(merge_output_tokens * 4, HARD_MAX_DOC_SUMMARY_CHARS),
            2_000,
            HARD_MAX_DOC_SUMMARY_CHARS,
        ),
        corpus_prompt_chars=env_int(
            "PAPER_STAGE_CORPUS_PROMPT_CHARS", 48_000, 10_000, HARD_MAX_CORPUS_PROMPT_CHARS
        ),
        corpus_summary_chars=env_int(
            "PAPER_STAGE_CORPUS_SUMMARY_CHARS",
            min(merge_output_tokens * 6, HARD_MAX_CORPUS_SUMMARY_CHARS),
            3_000,
            HARD_MAX_CORPUS_SUMMARY_CHARS,
        ),
        fallback_chars=env_int(
            "PAPER_STAGE_FALLBACK_CHARS", 40_000, 5_000, HARD_MAX_FALLBACK_CHARS
        ),
        paper_context_chars=env_int(
            "PAPER_CONTEXT_MAX_CHARS", 60_000, 10_000, HARD_MAX_PAPER_CONTEXT_CHARS
        ),
    )
    cache_identity = _runtime_cache_identity(
        config=config,
        chunk_size=chunk_size,
        max_chunks_per_doc=max_chunks_per_doc,
        max_output_tokens=max_output_tokens,
        merge_output_tokens=merge_output_tokens,
        budgets=budgets,
    )
    # Concurrency does not change model output, so it stays out of cache_identity.
    chunk_options = _ChunkStageOptions(
        topic=topic,
        instruction=instruction,
        cache_namespace=cache_namespace,
        cache_identity=cache_identity,
        config=config,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
        retries=retries,
        workers=workers,
        budgets=budgets,
    )

    doc_summaries: list[tuple[str, str]] = []
    coverage_notes: list[tuple[str, str]] = []
    total_chunk_count = 0

    for doc_index, document in enumerate(documents, start=1):
        source_budget = _available_chunk_source_chars(
            topic=topic,
            instruction=instruction,
            document_name=document.name,
            metadata_max_chars=budgets.metadata_chars,
            prompt_max_chars=budgets.chunk_prompt_chars,
        )
        effective_chunk_size = min(chunk_size, source_budget)
        all_chunks = split_text_into_chunks(document.content, effective_chunk_size)
        if not all_chunks:
            raise ValueError(f"文献 {document.name} 没有可分析的文本片段。")
        all_ranges = chunk_character_ranges(document.content, all_chunks)
        selected_chunks = select_chunk_coverage(all_chunks, max_chunks_per_doc)
        selected_indices = [index for index, _ in selected_chunks]
        selected_ranges = [all_ranges[index - 1] for index in selected_indices]
        selected_chars = sum(end - start + 1 for start, end in selected_ranges)
        provided_coverage = selected_chars / max(len(document.content), 1) * 100
        source_coverage = selected_chars / max(document.original_chars, 1) * 100
        source_scope = (
            f"解析得到 {document.original_chars} 字符；因长度上限已按首/中/尾采样后提供 "
            f"{document.extracted_chars} 字符。"
            if document.truncated
            else f"解析并提供 {document.original_chars} 字符。"
        )
        coverage_note = (
            f"> 覆盖说明：{source_scope}提供文本共切分为 {len(all_chunks)} 个片段，本次分析 "
            f"{len(selected_chunks)} 个；按字符约覆盖提供文本的 {provided_coverage:.0f}%"
            f"（约占解析文本的 {source_coverage:.0f}%）。选中范围："
            + "、".join(
                f"片段 {index}（字符 {start}-{end}）"
                for index, (start, end) in zip(selected_indices, selected_ranges, strict=True)
            )
            + "。"
        )
        coverage_notes.append((f"文献 {doc_index}：{document.name}", coverage_note))
        selection_note = ""
        if len(selected_chunks) < len(all_chunks):
            selection_note = "（已均匀覆盖开头、中部和结尾）"

        if progress:
            progress(
                f"开始分阶段阅读第 {doc_index}/{len(documents)} 篇文献：{document.name}，"
                f"原文共 {len(all_chunks)} 个片段，本次分析 {len(selected_chunks)} 个"
                f"{selection_note}。"
            )

        chunk_summaries = _summarize_document_chunks(
            document,
            options=chunk_options,
            all_chunks=all_chunks,
            all_ranges=all_ranges,
            selected_chunks=selected_chunks,
            progress=progress,
        )
        total_chunk_count += len(selected_chunks)

        doc_cache_path = _doc_summary_cache_path(
            document=document,
            topic=topic,
            instruction=instruction,
            chunk_summaries=chunk_summaries,
            cache_namespace=cache_namespace,
            cache_identity=cache_identity,
        )
        cached_doc_summary = _read_cache(doc_cache_path, cache_namespace=cache_namespace)
        if cached_doc_summary:
            if progress:
                progress(f"命中缓存：《{document.name}》单篇合并摘要已存在，跳过模型调用。")
            doc_summary = _truncate_text(cached_doc_summary, budgets.doc_summary_chars)
        else:
            if progress:
                progress(f"正在合并《{document.name}》的 {len(chunk_summaries)} 个片段摘要。")
            doc_summary_cacheable = True
            try:
                doc_summary = _call_model(
                    _doc_merge_prompt(
                        topic=topic,
                        instruction=instruction,
                        document_name=document.name,
                        chunk_summaries=chunk_summaries,
                        metadata_max_chars=budgets.metadata_chars,
                        prompt_max_chars=budgets.doc_prompt_chars,
                    ),
                    config=config,
                    max_output_tokens=merge_output_tokens,
                    timeout=timeout,
                    retries=retries,
                )
                doc_summary = _truncate_text(doc_summary, budgets.doc_summary_chars)
            except Exception as exc:
                doc_summary_cacheable = False
                fallback_enabled = env_bool("PAPER_FALLBACK_ON_MERGE_TIMEOUT", True)
                if not fallback_enabled:
                    raise
                if progress:
                    progress(
                        f"合并《{document.name}》时超时/失败，已保留片段缓存；"
                        "现在使用片段摘要拼接生成临时单篇摘要，避免前面工作白费。"
                    )
                doc_summary = _fallback_summary(
                    title=f"单篇文献临时摘要：{document.name}",
                    failure=exc,
                    sections=[
                        (f"片段摘要 {index}", summary)
                        for index, summary in enumerate(chunk_summaries, start=1)
                    ],
                    max_chars=min(budgets.fallback_chars, budgets.doc_summary_chars),
                )
            if doc_summary_cacheable:
                _write_cache(doc_cache_path, doc_summary, cache_namespace=cache_namespace)
        doc_summaries.append((document.name, f"{coverage_note}\n\n{doc_summary}"))

    if progress:
        progress(f"正在合并 {len(doc_summaries)} 篇文献的综合分析材料。")

    if len(doc_summaries) == 1:
        name, summary = doc_summaries[0]
        # Coverage is rendered separately below, so do not duplicate it in the
        # analysis section for the single-document case.
        doc_analysis = summary.removeprefix(f"{coverage_notes[0][1]}\n\n")
        final_analysis = f"### 文献：{name}\n{doc_analysis}"
    else:
        corpus_cache_path = _corpus_summary_cache_path(
            topic=topic,
            instruction=instruction,
            doc_summaries=doc_summaries,
            cache_namespace=cache_namespace,
            cache_identity=cache_identity,
        )
        cached_corpus_summary = _read_cache(corpus_cache_path, cache_namespace=cache_namespace)
        if cached_corpus_summary:
            if progress:
                progress("命中缓存：多篇文献综合分析材料已存在，跳过模型调用。")
            final_analysis = _truncate_text(cached_corpus_summary, budgets.corpus_summary_chars)
        else:
            corpus_summary_cacheable = True
            try:
                final_analysis = _call_model(
                    _corpus_merge_prompt(
                        topic=topic,
                        instruction=instruction,
                        doc_summaries=doc_summaries,
                        metadata_max_chars=budgets.metadata_chars,
                        prompt_max_chars=budgets.corpus_prompt_chars,
                    ),
                    config=config,
                    max_output_tokens=merge_output_tokens,
                    timeout=timeout,
                    retries=retries,
                )
                final_analysis = _truncate_text(final_analysis, budgets.corpus_summary_chars)
            except Exception as exc:
                corpus_summary_cacheable = False
                fallback_enabled = env_bool("PAPER_FALLBACK_ON_MERGE_TIMEOUT", True)
                if not fallback_enabled:
                    raise
                if progress:
                    progress("多篇综合合并超时/失败，使用单篇摘要拼接生成临时综合材料。")
                final_analysis = _fallback_summary(
                    title="多篇文献临时综合材料",
                    failure=exc,
                    sections=[
                        (f"文献 {index}：{name}", summary)
                        for index, (name, summary) in enumerate(doc_summaries, start=1)
                    ],
                    max_chars=min(budgets.fallback_chars, budgets.corpus_summary_chars),
                )
            if corpus_summary_cacheable:
                _write_cache(corpus_cache_path, final_analysis, cache_namespace=cache_namespace)

    final_context = _final_paper_context(
        coverage_notes=coverage_notes,
        analysis=final_analysis,
        max_chars=budgets.paper_context_chars,
    )

    return StagedLiteratureResult(
        paper_context=final_context,
        chunk_count=total_chunk_count,
        doc_count=len(documents),
    )
