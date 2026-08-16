from __future__ import annotations

import ipaddress
import os
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from auto_report_agent.settings import env_bool


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
        else (not is_public_deployment() or env_bool("APP_ALLOW_PRIVATE_API_HOSTS", False))
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
    """One request's LLM settings, passed explicitly instead of via os.environ."""

    api_key: str
    base_url: str
    model: str
    vision_model: str
    api_mode: str
    enable_web_search: bool

    @property
    def image_model(self) -> str:
        """Model to use for vision calls, falling back to the text model."""
        return self.vision_model or self.model

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


def llm_config_from_mapping(config: Mapping[str, object]) -> ResolvedLLMConfig:
    """Build a validated config object from the sidebar's session dictionary.

    Each browser session holds its own settings, so they are carried into the
    model calls as an argument. Writing them to ``os.environ`` instead would make
    one user's credentials process-global, which previously forced every run to
    serialize behind a single lock.
    """
    base_url = str(config.get("base_url") or "").strip()
    api_mode = str(config.get("api_mode") or "chat").strip().lower()
    return ResolvedLLMConfig(
        api_key=str(config.get("api_key") or "").strip(),
        base_url=validate_base_url(base_url) if base_url else "",
        model=str(config.get("model") or "").strip(),
        vision_model=str(config.get("vision_model") or "").strip(),
        api_mode="responses" if api_mode == "responses" else "chat",
        enable_web_search=bool(config.get("enable_web_search")) and api_mode == "responses",
    )


def redacted_api_summary(config: dict[str, str | bool]) -> str:
    base_url = str(config.get("base_url") or "").strip() or "未设置"
    model = str(config.get("model") or "").strip() or "未设置"
    api_mode = str(config.get("api_mode") or "chat")
    web_search = "开启" if config.get("enable_web_search") else "关闭"
    key_set = "已设置" if str(config.get("api_key") or "").strip() else "未设置"
    return f"Base URL：`{base_url}`；模型：`{model}`；Key：{key_set}；接口：{api_mode}；联网搜索：{web_search}"
