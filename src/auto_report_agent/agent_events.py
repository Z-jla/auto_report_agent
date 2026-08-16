"""Turn CrewAI / LangChain callback payloads into structured timeline events.

CrewAI hands ``step_callback`` and ``task_callback`` whatever object the current
agent loop produced, and the shape differs between versions and between the
LangChain and native paths. Parsing therefore works by class name with attribute
fallbacks, and an unrecognised payload still yields an event rather than being
dropped.

Kept separate from the Streamlit layer so it can be tested without a UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal


def safe_short_text(value: Any, max_len: int = 160) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


# ---------------------------------------------------------------------------
# Grok 风格事件 timeline
# ---------------------------------------------------------------------------


EventKind = Literal["system", "tool_call", "tool_result", "thought", "task_done"]


@dataclass
class AgentEvent:
    kind: EventKind
    title: str
    detail: str = ""
    icon: str = "•"
    agent: str = ""
    timestamp: str = ""


_AGENT_ICONS: dict[str, str] = {
    "researcher": "🔍",
    "writer": "✍️",
    "reviewer": "🧐",
    "paper_analyst": "📚",
    "专业研究员": "🔍",
    "专业报告写作者": "✍️",
    "内容审核与优化专家": "🧐",
    "学术文献分析专家": "📚",
}


def agent_icon(role: str) -> str:
    role_lower = (role or "").strip().lower()
    if not role_lower:
        return "🤖"
    for key, icon in _AGENT_ICONS.items():
        if key.lower() in role_lower:
            return icon
    return "🤖"


def parse_step_output(output: Any) -> AgentEvent:
    """把 CrewAI / langchain 抛给 step_callback 的对象解析成结构化事件。

    Always returns an event: an unrecognised payload becomes a generic system
    step naming its class, so a new CrewAI version cannot make the timeline go
    silent. Callers still guard the call itself, since a payload with a hostile
    ``__str__`` could raise.
    """
    cls = output.__class__.__name__
    agent_role = safe_short_text(
        getattr(output, "agent_role", "") or getattr(output, "agent", "") or "",
        40,
    )

    # langchain_core.agents.AgentAction：调用工具
    if cls in {"AgentAction", "ToolUsage", "ToolUsageStarted", "ToolUsageEvent"}:
        tool = getattr(output, "tool", "") or getattr(output, "tool_name", "")
        tool_input = getattr(output, "tool_input", "") or getattr(output, "input", "")
        log = (
            getattr(output, "log", "")
            or getattr(output, "thought", "")
            or getattr(output, "text", "")
        )
        title = f"调用工具 `{safe_short_text(tool, 40) or '未知工具'}`"
        if tool_input:
            title += f"：{safe_short_text(str(tool_input), 60)}"

        detail_parts: list[str] = []
        if log:
            detail_parts.append(f"**思考过程**\n\n{str(log).strip()}")
        if tool_input:
            tool_input_text = str(tool_input).strip()
            detail_parts.append(f"**调用参数**\n\n```\n{tool_input_text[:1500]}\n```")
        return AgentEvent(
            kind="tool_call",
            title=title,
            detail="\n\n".join(detail_parts),
            icon="🛠️",
            agent=agent_role,
        )

    # CrewAI ToolResult / ToolUsageFinished
    if cls in {"ToolResult", "ToolUsageFinished", "ToolUsageEnded"}:
        result = (
            getattr(output, "result", "")
            or getattr(output, "output", "")
            or getattr(output, "raw", "")
        )
        result_str = str(result)
        return AgentEvent(
            kind="tool_result",
            title=f"工具返回（约 {len(result_str)} 字符）",
            detail=result_str[:2000],
            icon="📥",
            agent=agent_role,
        )

    # langchain AgentFinish：阶段结论
    if cls == "AgentFinish":
        log = getattr(output, "log", "") or getattr(output, "text", "")
        return_values = getattr(output, "return_values", None) or {}
        output_text = ""
        if isinstance(return_values, dict):
            output_text = str(return_values.get("output") or "")
        return AgentEvent(
            kind="thought",
            title="得出阶段性结论",
            detail=(output_text or log or "").strip()[:2500],
            icon="💭",
            agent=agent_role,
        )

    # 兜底
    tool_name = getattr(output, "tool_name", "") or getattr(output, "tool", "")
    action = getattr(output, "action", "")
    if tool_name:
        return AgentEvent(
            kind="tool_call",
            title=f"使用工具 `{safe_short_text(tool_name, 40)}`",
            icon="🛠️",
            agent=agent_role,
        )
    if action:
        return AgentEvent(
            kind="system",
            title=f"执行动作：{safe_short_text(action, 80)}",
            agent=agent_role,
        )
    return AgentEvent(
        kind="system",
        title=f"完成执行步骤（{cls}）",
        agent=agent_role,
    )


def parse_task_output(output: Any) -> AgentEvent:
    name = safe_short_text(getattr(output, "name", ""), 60)
    agent = safe_short_text(getattr(output, "agent", ""), 40)
    raw = getattr(output, "raw", "") or getattr(output, "result", "") or ""
    raw_str = str(raw)

    parts = ["任务完成"]
    if name:
        parts.append(name)
    title = " · ".join(parts)
    if raw_str:
        title += f"（约 {len(raw_str)} 字符）"

    return AgentEvent(
        kind="task_done",
        title=title,
        detail=raw_str[:3000],
        icon="✅",
        agent=agent,
    )
