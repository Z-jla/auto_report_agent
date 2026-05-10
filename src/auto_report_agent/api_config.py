from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class ApiProviderPreset:
    name: str
    base_url: str
    model_hint: str
    api_mode: str = "chat"
    supports_web_search: bool = False


PROVIDER_PRESETS: dict[str, ApiProviderPreset] = {
    "自定义 OpenAI-compatible": ApiProviderPreset(
        name="自定义 OpenAI-compatible",
        base_url="",
        model_hint="填写你的模型名，例如 gpt-4o-mini / deepseek-chat / qwen-plus",
        api_mode="chat",
    ),
    "OpenAI": ApiProviderPreset(
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        model_hint="gpt-4o-mini / gpt-4.1-mini / gpt-5.5",
        api_mode="responses",
        supports_web_search=True,
    ),
    "DeepSeek": ApiProviderPreset(
        name="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        model_hint="deepseek-chat / deepseek-reasoner",
    ),
    "通义千问 DashScope": ApiProviderPreset(
        name="通义千问 DashScope",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_hint="qwen-plus / qwen-turbo / qwen-max",
    ),
    "月之暗面 Kimi": ApiProviderPreset(
        name="月之暗面 Kimi",
        base_url="https://api.moonshot.cn/v1",
        model_hint="moonshot-v1-8k / moonshot-v1-32k / kimi-k2",
    ),
    "智谱 GLM": ApiProviderPreset(
        name="智谱 GLM",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model_hint="glm-4-flash / glm-4-plus",
    ),
    "SiliconFlow": ApiProviderPreset(
        name="SiliconFlow",
        base_url="https://api.siliconflow.cn/v1",
        model_hint="Qwen/Qwen2.5-72B-Instruct / deepseek-ai/DeepSeek-V3",
    ),
    "OpenRouter": ApiProviderPreset(
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        model_hint="openai/gpt-4o-mini / anthropic/claude-3.5-sonnet / deepseek/deepseek-chat",
    ),
}


API_ENV_KEYS = {
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_VISION_MODEL",
    "LLM_API_MODE",
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL_NAME",
    "MODEL",
    "OPENAI_VISION_MODEL_NAME",
    "OPENAI_API_MODE",
    "ENABLE_WEB_SEARCH",
}


_API_ENV_LOCK = threading.RLock()


def default_api_config() -> dict[str, str | bool]:
    base_url = (
        os.getenv("LLM_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or ""
    )
    return {
        "provider": os.getenv("API_PROVIDER_NAME", "自定义 OpenAI-compatible"),
        "api_key": os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
        "base_url": base_url,
        "model": os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL_NAME", ""),
        "vision_model": os.getenv("LLM_VISION_MODEL") or os.getenv("OPENAI_VISION_MODEL_NAME", ""),
        "api_mode": os.getenv("LLM_API_MODE") or os.getenv("OPENAI_API_MODE", "chat"),
        "enable_web_search": os.getenv("ENABLE_WEB_SEARCH", "false").lower()
        in {"1", "true", "yes", "on"},
    }


def apply_api_config(config: dict[str, str | bool]) -> None:
    """Apply the current UI API config to environment variables used by CrewAI/helpers.

    Streamlit deployments should prefer session-only config. This function updates
    process env just-in-time because CrewAI and openai SDK helper calls read the
    canonical OPENAI_* variables. We also mirror to friendly LLM_* aliases so
    users scripting against either name see consistent values.
    """
    api_key = str(config.get("api_key") or "").strip()
    base_url = str(config.get("base_url") or "").strip().rstrip("/")
    model = str(config.get("model") or "").strip()
    vision_model = str(config.get("vision_model") or "").strip()
    api_mode = str(config.get("api_mode") or "chat").strip().lower()
    enable_web_search = bool(config.get("enable_web_search"))

    if api_key:
        os.environ["LLM_API_KEY"] = api_key
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["LLM_BASE_URL"] = base_url
        os.environ["OPENAI_API_BASE"] = base_url
        os.environ["OPENAI_BASE_URL"] = base_url
    if model:
        os.environ["LLM_MODEL"] = model
        os.environ["OPENAI_MODEL_NAME"] = model
        # Some CrewAI/LiteLLM paths read MODEL when no explicit llm is provided.
        os.environ["MODEL"] = model
    if vision_model:
        os.environ["LLM_VISION_MODEL"] = vision_model
        os.environ["OPENAI_VISION_MODEL_NAME"] = vision_model
    else:
        os.environ.pop("LLM_VISION_MODEL", None)
        os.environ.pop("OPENAI_VISION_MODEL_NAME", None)

    normalized_mode = "responses" if api_mode == "responses" else "chat"
    os.environ["LLM_API_MODE"] = normalized_mode
    os.environ["OPENAI_API_MODE"] = normalized_mode
    os.environ["ENABLE_WEB_SEARCH"] = "true" if enable_web_search else "false"


@contextmanager
def api_environment(config: dict[str, str | bool]) -> Iterator[None]:
    """Temporarily apply one user's API config for a model run.

    Streamlit sessions are per user, but ``os.environ`` is process-global.
    CrewAI and the openai SDK read the canonical OPENAI_* env vars, so this
    context serializes model runs, applies the current session config, and
    restores the previous env afterwards to avoid cross-user API key leakage.
    """
    with _API_ENV_LOCK:
        previous = {key: os.environ.get(key) for key in API_ENV_KEYS}
        apply_api_config(config)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def redacted_api_summary(config: dict[str, str | bool]) -> str:
    base_url = str(config.get("base_url") or "").strip() or "未设置"
    model = str(config.get("model") or "").strip() or "未设置"
    api_mode = str(config.get("api_mode") or "chat")
    web_search = "开启" if config.get("enable_web_search") else "关闭"
    key_set = "已设置" if str(config.get("api_key") or "").strip() else "未设置"
    return f"Base URL：`{base_url}`；模型：`{model}`；Key：{key_set}；接口：{api_mode}；联网搜索：{web_search}"
