from __future__ import annotations

import base64
import os
import re

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    DefaultHttpxClient,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from auto_report_agent.api_config import (
    ResolvedLLMConfig,
    resolve_llm_env,
    validate_base_url,
)
from auto_report_agent.settings import initialize_runtime

initialize_runtime()

DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024
SUPPORTED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}

_API_LABELS = {
    "chat": "Chat Completions",
    "responses": "Responses",
}
_ALTERNATE_API_STATUS_CODES = {405, 501}
_UNSUPPORTED_MARKERS = (
    "does not support",
    "doesn't support",
    "not support",
    "not supported",
    "unsupported",
    "not available",
    "unavailable",
    "not implemented",
    "method not allowed",
    "unknown endpoint",
    "unknown route",
    "no route",
)
_API_ROUTE_MARKERS = (
    "endpoint",
    "route",
    "api",
    "chat completion",
    "chat.completion",
    "chat/completion",
    "responses",
)
_RESOURCE_NOT_FOUND_MARKERS = (
    "model not found",
    "model does not exist",
    "model_not_found",
    "deployment not found",
    "deployment does not exist",
    "organization not found",
    "organization does not exist",
    "project not found",
    "project does not exist",
)
_NAMED_RESOURCE_MISSING_PATTERN = re.compile(
    r"\b(?:model|deployment|organization|project)\b"
    r"[^.;:{}\n]{0,160}"
    r"\b(?:(?:was|is)\s+)?(?:not found|does not exist)\b"
)


def validate_image_bytes(
    image_bytes: bytes,
    *,
    mime_type: str,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
) -> None:
    if not image_bytes:
        raise RuntimeError("图片内容为空。")
    if len(image_bytes) > max_bytes:
        raise ValueError(
            f"图片约 {len(image_bytes) / 1024 / 1024:.1f} MB，"
            f"超过上限 {max_bytes / 1024 / 1024:.1f} MB。"
        )
    normalized_mime = mime_type.lower().strip()
    if normalized_mime not in SUPPORTED_IMAGE_MIME_TYPES:
        raise ValueError(f"不支持的图片 MIME 类型：{mime_type}")

    signatures = {
        "image/png": image_bytes.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg": image_bytes.startswith(b"\xff\xd8\xff"),
        "image/webp": image_bytes.startswith(b"RIFF")
        and len(image_bytes) >= 12
        and image_bytes[8:12] == b"WEBP",
    }
    if not signatures[normalized_mime]:
        raise ValueError("图片内容与声明的 MIME 类型不一致。")


def _configured_max_image_bytes() -> int:
    raw = os.getenv("MAX_IMAGE_MB", "10").strip()
    try:
        megabytes = int(raw)
    except ValueError:
        megabytes = 10
    return max(1, min(50, megabytes)) * 1024 * 1024


def _chat_vision_request(client: OpenAI, *, model: str, prompt: str, data_url: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=1600,
    )
    content = response.choices[0].message.content
    if isinstance(content, str) and content.strip():
        return content.strip()
    raise RuntimeError("图片识别失败：模型在 Chat Completions 接口返回空内容。")


def _responses_vision_request(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    data_url: str,
) -> str:
    response = client.responses.create(
        model=model,
        # The SDK types this as a union of overloaded TypedDicts that a
        # dict literal cannot match, though the payload is correct.
        input=[
            {  # type: ignore[misc, list-item]
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        max_output_tokens=1600,
    )
    content = getattr(response, "output_text", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    raise RuntimeError("图片识别失败：模型在 Responses 接口返回空内容。")


def _api_error_text(error: APIStatusError) -> str:
    parts = [str(error)]
    body = getattr(error, "body", None)
    if body is not None:
        parts.append(str(body))
    return " ".join(parts).lower()


def _should_try_alternate_api(error: APIStatusError) -> bool:
    """Return whether a status error clearly points to an API/capability mismatch."""
    status_code = getattr(error, "status_code", None)
    if status_code in _ALTERNATE_API_STATUS_CODES:
        return True

    details = _api_error_text(error)
    if status_code == 404:
        # Keep the resource and its missing reason in the same short phrase.
        # Merely finding both words anywhere in the error misclassifies explicit
        # route errors such as "Responses endpoint not found; model X uses Chat".
        named_resource_missing = _NAMED_RESOURCE_MISSING_PATTERN.search(details) is not None
        if named_resource_missing or any(
            marker in details for marker in _RESOURCE_NOT_FOUND_MARKERS
        ):
            return False
        # A generic 404 from an OpenAI-compatible server usually means that the
        # selected API route is absent. More specific model/deployment failures
        # are deliberately excluded above because changing APIs cannot fix them.
        return "not found" in details or any(marker in details for marker in _API_ROUTE_MARKERS)

    if status_code not in {400, 415, 422}:
        return False
    return any(marker in details for marker in _UNSUPPORTED_MARKERS) and any(
        marker in details for marker in _API_ROUTE_MARKERS
    )


def _request_vision(
    client: OpenAI,
    *,
    api_mode: str,
    model: str,
    prompt: str,
    data_url: str,
) -> str:
    if api_mode == "responses":
        return _responses_vision_request(
            client,
            model=model,
            prompt=prompt,
            data_url=data_url,
        )
    return _chat_vision_request(client, model=model, prompt=prompt, data_url=data_url)


def _friendly_operational_error(error: Exception) -> RuntimeError:
    if isinstance(error, (AuthenticationError, PermissionDeniedError)):
        return RuntimeError(
            f"图片识别失败：API Key 鉴权未通过（{error.__class__.__name__}）。"
            " 请检查 LLM_API_KEY / LLM_BASE_URL（或 OPENAI_API_KEY / OPENAI_API_BASE）是否匹配。"
        )
    if isinstance(error, (APIConnectionError, APITimeoutError)):
        return RuntimeError(
            f"图片识别失败：网络连接异常（{error.__class__.__name__}）。 请检查网络或反向代理。"
        )
    return RuntimeError("图片识别失败：触发限流（429）。请稍后重试或更换模型/账号。")


def _run_vision_request(
    client: OpenAI,
    *,
    api_mode: str,
    model: str,
    prompt: str,
    data_url: str,
) -> str:
    alternate_mode = "chat" if api_mode == "responses" else "responses"
    try:
        return _request_vision(
            client,
            api_mode=api_mode,
            model=model,
            prompt=prompt,
            data_url=data_url,
        )
    except (
        AuthenticationError,
        PermissionDeniedError,
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
    ) as error:
        raise _friendly_operational_error(error) from error
    except APIStatusError as primary_error:
        if not _should_try_alternate_api(primary_error):
            status_code = getattr(primary_error, "status_code", "未知")
            raise RuntimeError(
                f"图片识别失败：{_API_LABELS[api_mode]} 接口返回 HTTP {status_code}。"
                f" {primary_error}"
            ) from primary_error

        try:
            return _request_vision(
                client,
                api_mode=alternate_mode,
                model=model,
                prompt=prompt,
                data_url=data_url,
            )
        except (
            AuthenticationError,
            PermissionDeniedError,
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
        ) as error:
            raise _friendly_operational_error(error) from error
        except APIStatusError as alternate_error:
            raise RuntimeError(
                "图片识别失败：当前模型或兼容接口可能不支持视觉输入。"
                f" {_API_LABELS[api_mode]} 错误：{primary_error};"
                f" {_API_LABELS[alternate_mode]} 错误：{alternate_error}"
            ) from alternate_error


def analyze_image_content(
    image_bytes: bytes,
    mime_type: str = "image/png",
    instruction: str = "",
    *,
    config: ResolvedLLMConfig | None = None,
) -> str:
    """Use the configured OpenAI-compatible model to analyze an uploaded image.

    Returns a Markdown text containing OCR result, visual description, key facts,
    and suggested next-step task interpretation.

    ``config`` carries one caller's settings; it defaults to the process
    environment, which is what the CLI uses.
    """
    env = config or resolve_llm_env(prefer_vision_model=True)
    model = env.image_model

    if not env.api_key:
        raise RuntimeError("缺少 LLM_API_KEY（或 OPENAI_API_KEY），无法进行图片识别。")
    if not env.base_url:
        raise RuntimeError("缺少 LLM_BASE_URL（或 OPENAI_API_BASE），无法进行图片识别。")
    if not model:
        raise RuntimeError(
            "缺少视觉模型：请设置 LLM_VISION_MODEL 或 LLM_MODEL（或其 OPENAI_* 对应名）。"
        )
    validate_image_bytes(
        image_bytes,
        mime_type=mime_type,
        max_bytes=_configured_max_image_bytes(),
    )

    try:
        base_url = validate_base_url(env.base_url)
    except ValueError as exc:
        raise RuntimeError(f"Base URL 安全校验未通过：{exc}") from exc

    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    prompt = f"""
你是一个严谨的多模态信息分析助手。请识别用户上传图片中的内容，并输出结构化 Markdown。

请完成：
1. OCR：尽可能完整提取图片中的文字；
2. 图像理解：说明图片类型、主体、界面/图表/文档结构；
3. 关键信息：列出可用于后续研究或写作的事实、数字、实体、任务要求；
4. 后续任务建议：根据图片内容判断用户可能希望系统完成什么工作；
5. 如果图片信息不足或不清晰，请明确说明。

用户补充指令：{instruction or "无"}
""".strip()

    http_client = DefaultHttpxClient(follow_redirects=False)
    client: OpenAI | None = None
    try:
        client = OpenAI(
            api_key=env.api_key,
            base_url=base_url,
            http_client=http_client,
            timeout=90.0,
            max_retries=1,
        )
        api_mode = "responses" if env.api_mode == "responses" else "chat"
        return _run_vision_request(
            client,
            api_mode=api_mode,
            model=model,
            prompt=prompt,
            data_url=data_url,
        )
    finally:
        # OpenAI.close() owns and closes the supplied HTTP client. If OpenAI's
        # constructor itself fails, close that client directly instead.
        if client is None:
            http_client.close()
        else:
            client.close()
