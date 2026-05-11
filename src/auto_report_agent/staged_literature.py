from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from auto_report_agent.api_config import resolve_llm_env
from auto_report_agent.document_ingest import ParsedDocument
from auto_report_agent.openai_web_search_tool import OpenAIWebSearchTool
from auto_report_agent.settings import OUTPUT_DIR, initialize_runtime

ProgressCallback = Callable[[str], None]
CACHE_VERSION = "staged-literature-v2"
initialize_runtime()
CACHE_DIR = OUTPUT_DIR / "literature_cache"


@dataclass
class StagedLiteratureResult:
    paper_context: str
    chunk_count: int
    doc_count: int


def _get_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _get_bool_env(name: str, default: bool = True) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off", "禁用", "关闭"}


def _hash_text(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:length]


def _safe_filename(name: str, max_len: int = 80) -> str:
    stem = Path(name).stem or "document"
    safe = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", stem).strip("._-")
    return (safe or "document")[:max_len]


def _cache_root_for_document(document: ParsedDocument) -> Path:
    doc_hash = _hash_text(f"{CACHE_VERSION}\n{document.name}\n{document.content}", 16)
    return CACHE_DIR / f"{_safe_filename(document.name)}_{doc_hash}"


def _read_cache(path: Path) -> str | None:
    if not _get_bool_env("PAPER_CACHE_ENABLED", True):
        return None
    try:
        if path.exists():
            content = path.read_text(encoding="utf-8").strip()
            return content or None
    except OSError:
        return None
    return None


def _write_cache(path: Path, content: str) -> None:
    if not _get_bool_env("PAPER_CACHE_ENABLED", True):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)


def _chunk_cache_path(
    *,
    document: ParsedDocument,
    topic: str,
    instruction: str,
    chunk_index: int,
    total_chunks: int,
    chunk_text: str,
) -> Path:
    key = _hash_text(
        json.dumps(
            {
                "version": CACHE_VERSION,
                "kind": "chunk",
                "document": document.name,
                "topic": topic,
                "instruction": instruction,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "chunk_text": chunk_text,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        20,
    )
    return _cache_root_for_document(document) / "chunks" / f"chunk_{chunk_index:03d}_{key}.md"


def _doc_summary_cache_path(
    *,
    document: ParsedDocument,
    topic: str,
    instruction: str,
    chunk_summaries: Sequence[str],
) -> Path:
    key = _hash_text(
        json.dumps(
            {
                "version": CACHE_VERSION,
                "kind": "doc_summary",
                "document": document.name,
                "topic": topic,
                "instruction": instruction,
                "chunk_summaries": list(chunk_summaries),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        20,
    )
    return _cache_root_for_document(document) / f"doc_summary_{key}.md"


def _corpus_summary_cache_path(
    *,
    topic: str,
    instruction: str,
    doc_summaries: Sequence[tuple[str, str]],
) -> Path:
    key = _hash_text(
        json.dumps(
            {
                "version": CACHE_VERSION,
                "kind": "corpus_summary",
                "topic": topic,
                "instruction": instruction,
                "doc_summaries": list(doc_summaries),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        20,
    )
    return CACHE_DIR / "corpus" / f"corpus_summary_{key}.md"


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


def _call_model(prompt: str, *, max_output_tokens: int, timeout: int, retries: int = 1) -> str:
    env = resolve_llm_env()

    if not env.api_key:
        raise RuntimeError("缺少 LLM_API_KEY（或 OPENAI_API_KEY），无法进行分阶段文献分析。")
    if not env.base_url:
        raise RuntimeError("缺少 LLM_BASE_URL（或 OPENAI_API_BASE），无法进行分阶段文献分析。")
    if not env.model:
        raise RuntimeError("缺少 LLM_MODEL（或 OPENAI_MODEL_NAME），无法进行分阶段文献分析。")

    if env.api_mode == "responses":
        endpoint = f"{env.base_url}/responses"
        payload = {
            "model": env.model,
            "input": prompt,
            "store": False,
            "max_output_tokens": max_output_tokens,
        }
    else:
        endpoint = f"{env.base_url}/chat/completions"
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
        "User-Agent": "auto-report-agent/0.1 staged-literature",
    }

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            endpoint,
            data=data,
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw)
            text = _extract_response_text(parsed)
            if text:
                return text
            raise RuntimeError("模型接口返回成功，但没有解析到文本内容。")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {body[:1200]}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc

        if attempt < retries:
            time.sleep(min(8, 2 + attempt * 3))

    raise RuntimeError(f"分阶段文献分析调用模型失败：{type(last_error).__name__}: {last_error}")


def _chunk_prompt(
    *,
    topic: str,
    instruction: str,
    document_name: str,
    chunk_index: int,
    total_chunks: int,
    chunk_text: str,
) -> str:
    return f"""
你是严谨的学术文献阅读助手。请只基于下面这一段文献内容做局部摘要，不要编造文中没有的信息。

文献主题/综述方向：{topic or '未指定'}
用户补充要求：{instruction or '无'}
文献文件：{document_name}
当前片段：第 {chunk_index}/{total_chunks} 段

请输出 Markdown，包含：
1. 本片段核心内容
2. 涉及的研究问题/背景
3. 方法、模型、数据或实验信息
4. 关键结论或发现
5. 创新点/局限性/待核实信息
6. 可引用的事实依据

如果某项未出现，请写“本片段未明确说明”。

## 文献片段
{chunk_text}
""".strip()


def _doc_merge_prompt(*, topic: str, instruction: str, document_name: str, chunk_summaries: Sequence[str]) -> str:
    joined = "\n\n".join(
        f"## 片段摘要 {index}\n{summary}" for index, summary in enumerate(chunk_summaries, start=1)
    )
    return f"""
你是学术文献分析专家。下面是同一篇文献的多个片段摘要。请合并为一份“单篇文献结构化摘要”。

文献主题/综述方向：{topic or '未指定'}
用户补充要求：{instruction or '无'}
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
{joined}
""".strip()


def _corpus_merge_prompt(*, topic: str, instruction: str, doc_summaries: Sequence[tuple[str, str]]) -> str:
    joined = "\n\n".join(
        f"## 文献 {index}：{name}\n{summary}" for index, (name, summary) in enumerate(doc_summaries, start=1)
    )
    return f"""
你是中文文献综述助手。下面是多篇文献的单篇结构化摘要。请生成给后续写作 Agent 使用的“综合文献分析材料”。

文献主题/综述方向：{topic or '未指定'}
用户补充要求：{instruction or '无'}

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
{joined}
""".strip()


def summarize_documents_staged(
    documents: Sequence[ParsedDocument],
    *,
    topic: str,
    instruction: str,
    progress: ProgressCallback | None = None,
) -> StagedLiteratureResult:
    """分块阅读文献，生成压缩后的综合文献上下文。"""
    chunk_size = _get_int_env("PAPER_CHUNK_SIZE", 6000, 2000, 12000)
    max_chunks_per_doc = _get_int_env("PAPER_MAX_CHUNKS_PER_DOC", 10, 1, 30)
    max_output_tokens = _get_int_env("PAPER_STAGE_MAX_OUTPUT_TOKENS", 900, 300, 2000)
    merge_output_tokens = _get_int_env("PAPER_STAGE_MERGE_TOKENS", 1400, 500, 3000)
    timeout = _get_int_env("PAPER_STAGE_TIMEOUT", 90, 20, 180)
    retries = _get_int_env("PAPER_STAGE_RETRIES", 1, 0, 3)

    doc_summaries: list[tuple[str, str]] = []
    total_chunk_count = 0

    for doc_index, document in enumerate(documents, start=1):
        chunks = split_text_into_chunks(document.content, chunk_size)
        if len(chunks) > max_chunks_per_doc:
            chunks = chunks[:max_chunks_per_doc]
            truncated_note = f"（原文切出更多片段，已按上限只分析前 {max_chunks_per_doc} 段）"
        else:
            truncated_note = ""

        if progress:
            progress(
                f"开始分阶段阅读第 {doc_index}/{len(documents)} 篇文献：{document.name}，"
                f"共 {len(chunks)} 个片段{truncated_note}。"
            )

        chunk_summaries: list[str] = []
        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk_cache_path = _chunk_cache_path(
                document=document,
                topic=topic,
                instruction=instruction,
                chunk_index=chunk_index,
                total_chunks=len(chunks),
                chunk_text=chunk,
            )
            cached_summary = _read_cache(chunk_cache_path)
            if cached_summary:
                if progress:
                    progress(f"命中缓存：《{document.name}》第 {chunk_index}/{len(chunks)} 个片段已完成，跳过模型调用。")
                chunk_summaries.append(cached_summary)
                total_chunk_count += 1
                continue

            if progress:
                progress(f"正在分析《{document.name}》第 {chunk_index}/{len(chunks)} 个片段。")
            summary = _call_model(
                _chunk_prompt(
                    topic=topic,
                    instruction=instruction,
                    document_name=document.name,
                    chunk_index=chunk_index,
                    total_chunks=len(chunks),
                    chunk_text=chunk,
                ),
                max_output_tokens=max_output_tokens,
                timeout=timeout,
                retries=retries,
            )
            _write_cache(chunk_cache_path, summary)
            chunk_summaries.append(summary)
            total_chunk_count += 1

        doc_cache_path = _doc_summary_cache_path(
            document=document,
            topic=topic,
            instruction=instruction,
            chunk_summaries=chunk_summaries,
        )
        cached_doc_summary = _read_cache(doc_cache_path)
        if cached_doc_summary:
            if progress:
                progress(f"命中缓存：《{document.name}》单篇合并摘要已存在，跳过模型调用。")
            doc_summary = cached_doc_summary
        else:
            if progress:
                progress(f"正在合并《{document.name}》的 {len(chunk_summaries)} 个片段摘要。")
            try:
                doc_summary = _call_model(
                    _doc_merge_prompt(
                        topic=topic,
                        instruction=instruction,
                        document_name=document.name,
                        chunk_summaries=chunk_summaries,
                    ),
                    max_output_tokens=merge_output_tokens,
                    timeout=timeout,
                    retries=retries,
                )
            except Exception as exc:
                fallback_enabled = _get_bool_env("PAPER_FALLBACK_ON_MERGE_TIMEOUT", True)
                if not fallback_enabled:
                    raise
                if progress:
                    progress(
                        f"合并《{document.name}》时超时/失败，已保留片段缓存；"
                        "现在使用片段摘要拼接生成临时单篇摘要，避免前面工作白费。"
                    )
                joined = "\n\n".join(
                    f"## 片段摘要 {index}\n{summary}"
                    for index, summary in enumerate(chunk_summaries, start=1)
                )
                doc_summary = (
                    f"# 单篇文献临时摘要：{document.name}\n\n"
                    f"> 自动合并步骤失败，以下内容由已成功缓存的片段摘要拼接而成。"
                    f"失败原因：{type(exc).__name__}: {exc}\n\n"
                    f"{joined}"
                )
            _write_cache(doc_cache_path, doc_summary)
        doc_summaries.append((document.name, doc_summary))

    if progress:
        progress(f"正在合并 {len(doc_summaries)} 篇文献的综合分析材料。")

    if len(doc_summaries) == 1:
        name, summary = doc_summaries[0]
        final_context = f"# 分阶段文献分析结果\n\n## 文献：{name}\n{summary}"
    else:
        corpus_cache_path = _corpus_summary_cache_path(
            topic=topic,
            instruction=instruction,
            doc_summaries=doc_summaries,
        )
        cached_corpus_summary = _read_cache(corpus_cache_path)
        if cached_corpus_summary:
            if progress:
                progress("命中缓存：多篇文献综合分析材料已存在，跳过模型调用。")
            final_context = cached_corpus_summary
        else:
            try:
                final_context = _call_model(
                    _corpus_merge_prompt(
                        topic=topic,
                        instruction=instruction,
                        doc_summaries=doc_summaries,
                    ),
                    max_output_tokens=merge_output_tokens,
                    timeout=timeout,
                    retries=retries,
                )
            except Exception as exc:
                fallback_enabled = _get_bool_env("PAPER_FALLBACK_ON_MERGE_TIMEOUT", True)
                if not fallback_enabled:
                    raise
                if progress:
                    progress("多篇综合合并超时/失败，使用单篇摘要拼接生成临时综合材料。")
                joined = "\n\n".join(
                    f"## 文献 {index}：{name}\n{summary}"
                    for index, (name, summary) in enumerate(doc_summaries, start=1)
                )
                final_context = (
                    "# 多篇文献临时综合材料\n\n"
                    f"> 自动综合步骤失败，以下内容由已成功缓存的单篇摘要拼接而成。"
                    f"失败原因：{type(exc).__name__}: {exc}\n\n"
                    f"{joined}"
                )
            _write_cache(corpus_cache_path, final_context)

    return StagedLiteratureResult(
        paper_context=final_context,
        chunk_count=total_chunk_count,
        doc_count=len(documents),
    )
