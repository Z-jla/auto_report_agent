import json
import urllib.error
import urllib.request
from typing import Any

from crewai.tools import BaseTool

from auto_report_agent.api_config import resolve_llm_env
from auto_report_agent.settings import initialize_runtime

initialize_runtime()


class OpenAIWebSearchTool(BaseTool):
    name: str = "openai_web_search"
    description: str = (
        "使用 OpenAI 兼容商的 Responses API web_search_preview 工具联网搜索最新资料。"
        "输入 query，返回搜索后的摘要、关键发现和可能的引用信息。"
    )

    def _run(self, query: str) -> str:
        env = resolve_llm_env()

        if not env.api_key:
            return "搜索失败：缺少 LLM_API_KEY（或 OPENAI_API_KEY）。"
        if not env.base_url:
            return "搜索失败：缺少 LLM_BASE_URL（或 OPENAI_API_BASE）。"
        if not env.model:
            return "搜索失败：缺少 LLM_MODEL（或 OPENAI_MODEL_NAME）。"
        if env.api_mode != "responses" or not env.enable_web_search:
            return (
                "已跳过联网搜索：当前 API 配置未启用 Responses API web_search_preview。"
                "如果你的服务商支持 Responses 工具调用，请在侧边栏把后端直连接口改为 responses，"
                "并勾选“启用 Responses web_search_preview 联网搜索”；否则请使用文献上传或在主题中提供资料来源。"
            )

        payload = {
            "model": env.model,
            "input": (
                "请联网搜索并整理资料。要求：\n"
                "1. 优先使用最新、可信的信息；\n"
                "2. 输出核心发现、重要事实、来源标题/URL；\n"
                "3. 如果信息不确定，请明确说明。\n\n"
                f"搜索主题：{query}"
            ),
            "tools": [{"type": "web_search_preview"}],
            "store": False,
            "max_output_tokens": 1600,
        }

        request = urllib.request.Request(
            f"{env.base_url}/responses",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {env.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "auto-report-agent/0.1",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                raw = response.read().decode("utf-8", errors="replace")
                data = json.loads(raw)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return f"搜索失败：HTTP {exc.code}，响应：{body[:1000]}"
        except Exception as exc:
            return f"搜索失败：{type(exc).__name__}: {exc}"

        text = self._extract_response_text(data)
        if text:
            return text

        return "搜索完成，但没有解析到文本结果。原始响应片段：\n" + json.dumps(
            data, ensure_ascii=False
        )[:2000]

    @staticmethod
    def _extract_response_text(data: dict[str, Any]) -> str:
        if isinstance(data.get("output_text"), str) and data["output_text"].strip():
            return data["output_text"].strip()

        chunks: list[str] = []
        for item in data.get("output", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "web_search_call":
                action = item.get("action") or {}
                query = action.get("query") or action.get("queries")
                if query:
                    chunks.append(f"[Web search] {query}")
                continue

            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                if content.get("type") in {"output_text", "text"}:
                    text = content.get("text", "")
                    if text:
                        chunks.append(text)

                    annotations = content.get("annotations") or []
                    sources = []
                    for ann in annotations:
                        if not isinstance(ann, dict):
                            continue
                        title = ann.get("title") or ann.get("url") or "source"
                        url = ann.get("url")
                        if url:
                            sources.append(f"- {title}: {url}")
                    if sources:
                        chunks.append("参考来源：\n" + "\n".join(sources))

        return "\n\n".join(chunks).strip()
