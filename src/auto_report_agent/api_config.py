from __future__ import annotations

import ipaddress
import os
import socket
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator
from urllib.parse import urlsplit


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

DEFAULT_PUBLIC_API_HOSTS = frozenset(
    host
    for preset in PROVIDER_PRESETS.values()
    if preset.base_url and (host := urlsplit(preset.base_url).hostname)
)


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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def is_public_deployment() -> bool:
    """Return whether the app should apply public/multi-user safety defaults."""
    return os.getenv("APP_DEPLOYMENT_MODE", "public").strip().lower() != "local"


def _is_disallowed_address(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return any(
        (
            not ip.is_global,
            ip.is_private,
            ip.is_loopback,
            ip.is_link_local,
            ip.is_multicast,
            ip.is_reserved,
            ip.is_unspecified,
        )
    )


def _host_matches(host: str, patterns: set[str] | frozenset[str]) -> bool:
    return any(
        host == pattern
        or (pattern.startswith("*.") and host.endswith(pattern[1:]) and host != pattern[2:])
        for pattern in patterns
    )


def validate_base_url(
    base_url: str,
    *,
    allow_private: bool | None = None,
    resolve_dns: bool = True,
) -> str:
    """Validate an API root URL and reject common SSRF targets.

    Private/local targets are allowed only in explicit local deployment mode or
    when ``APP_ALLOW_PRIVATE_API_HOSTS=true`` is set. Public deployments require
    HTTPS and resolve hostnames before a model request to block private addresses.
    """
    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized:
        raise ValueError("Base URL 不能为空。")

    try:
        parsed = urlsplit(normalized)
        host_value = parsed.hostname
    except ValueError as exc:
        raise ValueError("Base URL 格式无效。") from exc
    if parsed.scheme not in {"http", "https"} or not host_value:
        raise ValueError("Base URL 必须是完整的 http(s) URL。")
    if parsed.username or parsed.password:
        raise ValueError("Base URL 不能包含用户名或密码。")
    if parsed.query or parsed.fragment:
        raise ValueError("Base URL 不能包含查询参数或 URL 片段。")
    try:
        parsed_port = parsed.port
    except ValueError as exc:
        raise ValueError("Base URL 端口无效。") from exc

    private_allowed = (
        allow_private
        if allow_private is not None
        else (not is_public_deployment() or _env_bool("APP_ALLOW_PRIVATE_API_HOSTS"))
    )
    if not private_allowed and parsed.scheme != "https":
        raise ValueError("公共部署只允许 HTTPS API 地址。")

    host = host_value.rstrip(".").lower()
    if not private_allowed and (host == "localhost" or host.endswith((".localhost", ".local"))):
        raise ValueError("公共部署不允许访问本机或内网 API 地址。")

    configured_hosts = {
        value.strip().lower()
        for value in os.getenv("APP_ALLOWED_API_HOSTS", "").split(",")
        if value.strip()
    }
    if is_public_deployment() and allow_private is not True:
        allowed_hosts = DEFAULT_PUBLIC_API_HOSTS | configured_hosts
        if not _host_matches(host, allowed_hosts):
            raise ValueError(
                "公共部署只允许内置服务商或 APP_ALLOWED_API_HOSTS 明确列出的 API 主机。"
            )
    elif configured_hosts and not _host_matches(host, configured_hosts):
        raise ValueError("该 API 主机不在 APP_ALLOWED_API_HOSTS 白名单中。")

    if not private_allowed:
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        if literal is not None and _is_disallowed_address(str(literal)):
            raise ValueError("公共部署不允许访问本机、私网或保留地址。")

        if resolve_dns:
            try:
                addresses = {
                    item[4][0]
                    for item in socket.getaddrinfo(
                        host, parsed_port or 443, type=socket.SOCK_STREAM
                    )
                }
            except socket.gaierror as exc:
                raise ValueError(f"无法解析 API 主机：{host}") from exc
            if not addresses or any(_is_disallowed_address(address) for address in addresses):
                raise ValueError("API 主机解析到了本机、私网或保留地址，已拒绝请求。")

    return normalized


def _first_env(*names: str) -> str:
    """Return the first non-empty environment variable among ``names``, stripped."""
    for name in names:
        value = os.getenv(name, "")
        if value and value.strip():
            return value.strip()
    return ""


@dataclass(frozen=True)
class ResolvedLLMConfig:
    """Runtime view of the active LLM environment, with LLM_* taking priority over OPENAI_*."""

    api_key: str
    base_url: str
    model: str
    vision_model: str
    api_mode: str
    enable_web_search: bool

    def require(self, *fields: str) -> None:
        """Raise ``RuntimeError`` when any requested field is empty."""
        labels = {
            "api_key": "LLM_API_KEY（或 OPENAI_API_KEY）",
            "base_url": "LLM_BASE_URL（或 OPENAI_API_BASE）",
            "model": "LLM_MODEL（或 OPENAI_MODEL_NAME）",
            "vision_model": "LLM_VISION_MODEL（或 OPENAI_VISION_MODEL_NAME / LLM_MODEL）",
        }
        missing = [labels[field] for field in fields if not getattr(self, field, "")]
        if missing:
            raise RuntimeError("缺少环境变量：" + "、".join(missing))


def resolve_llm_env(*, prefer_vision_model: bool = False) -> ResolvedLLMConfig:
    """Read the current LLM-related environment into a single config object.

    ``prefer_vision_model=True`` makes ``model`` fall back through the vision-model
    aliases first (useful for image analysis). ``api_mode`` is normalized to
    ``"chat"`` or ``"responses"``. No default model name is invented — callers must
    check ``ResolvedLLMConfig.require(...)`` (or use the raised error directly) to
    surface a clean message when the user forgot to configure a model.
    """
    api_key = _first_env("LLM_API_KEY", "OPENAI_API_KEY")
    base_url = _first_env("LLM_BASE_URL", "OPENAI_API_BASE", "OPENAI_BASE_URL").rstrip("/")
    text_model = _first_env("LLM_MODEL", "OPENAI_MODEL_NAME")
    vision_model = _first_env("LLM_VISION_MODEL", "OPENAI_VISION_MODEL_NAME")

    if prefer_vision_model:
        model = vision_model or text_model
    else:
        model = text_model

    raw_mode = _first_env("LLM_API_MODE", "OPENAI_API_MODE") or "chat"
    api_mode = "responses" if raw_mode.lower() == "responses" else "chat"

    enable_web_search = os.getenv("ENABLE_WEB_SEARCH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    return ResolvedLLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        vision_model=vision_model,
        api_mode=api_mode,
        enable_web_search=enable_web_search,
    )


def default_api_config(*, include_server_values: bool = True) -> dict[str, str | bool]:
    resolved = resolve_llm_env()
    return {
        "provider": (
            os.getenv("API_PROVIDER_NAME", "自定义 OpenAI-compatible")
            if include_server_values
            else "自定义 OpenAI-compatible"
        ),
        "api_key": resolved.api_key if include_server_values else "",
        "base_url": resolved.base_url if include_server_values else "",
        "model": resolved.model if include_server_values else "",
        "vision_model": resolved.vision_model if include_server_values else "",
        "api_mode": resolved.api_mode if include_server_values else "chat",
        "enable_web_search": resolved.enable_web_search if include_server_values else False,
    }


def apply_api_config(config: dict[str, str | bool]) -> None:
    """Apply the current UI API config to environment variables used by CrewAI/helpers.

    Streamlit deployments should prefer session-only config. This function updates
    process env just-in-time because CrewAI and openai SDK helper calls read the
    canonical OPENAI_* variables. We also mirror to friendly LLM_* aliases so
    users scripting against either name see consistent values.
    """
    api_key = str(config.get("api_key") or "").strip()
    raw_base_url = str(config.get("base_url") or "").strip()
    base_url = validate_base_url(raw_base_url) if raw_base_url else ""
    model = str(config.get("model") or "").strip()
    vision_model = str(config.get("vision_model") or "").strip()
    api_mode = str(config.get("api_mode") or "chat").strip().lower()
    enable_web_search = bool(config.get("enable_web_search"))

    # Prevent an empty per-session value from inheriting another session's or
    # the server process's credential/configuration.
    for key in API_ENV_KEYS:
        os.environ.pop(key, None)

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
        try:
            apply_api_config(config)
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
