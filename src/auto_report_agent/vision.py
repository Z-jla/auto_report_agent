from __future__ import annotations

import base64
import os

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    DefaultHttpxClient,
    NotFoundError,
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
    client = OpenAI(
        api_key=env.api_key,
        base_url=base_url,
        http_client=DefaultHttpxClient(follow_redirects=False),
        timeout=90.0,
        max_retries=1,
    )

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

    try:
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
        if content:
            return content.strip()
        raise RuntimeError("图片识别失败：模型在 Chat Completions 接口返回空内容。")
    except (AuthenticationError, PermissionDeniedError) as exc:
        raise RuntimeError(
            f"图片识别失败：API Key 鉴权未通过（{exc.__class__.__name__}）。"
            " 请检查 LLM_API_KEY / LLM_BASE_URL（或 OPENAI_API_KEY / OPENAI_API_BASE）是否匹配。"
        ) from exc
    except (APIConnectionError, APITimeoutError) as exc:
        raise RuntimeError(
            f"图片识别失败：网络连接异常（{exc.__class__.__name__}）。 请检查网络或反向代理。"
        ) from exc
    except RateLimitError as exc:
        raise RuntimeError("图片识别失败：触发限流（429）。请稍后重试或更换模型/账号。") from exc
    except (BadRequestError, NotFoundError) as chat_error:
        # Chat Completions 拒绝请求或路径不存在，多半是兼容商把视觉接口放在 Responses API 上。
        try:
            fallback = client.responses.create(
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
        except (AuthenticationError, PermissionDeniedError) as exc:
            raise RuntimeError(
                f"图片识别失败：API Key 鉴权未通过（{exc.__class__.__name__}）。"
            ) from exc
        except (APIConnectionError, APITimeoutError) as exc:
            raise RuntimeError(f"图片识别失败：网络连接异常（{exc.__class__.__name__}）。") from exc
        except RateLimitError as exc:
            raise RuntimeError("图片识别失败：触发限流（429）。") from exc
        except (BadRequestError, NotFoundError) as responses_error:
            raise RuntimeError(
                "图片识别失败：当前模型或兼容接口可能不支持视觉输入。"
                f" Chat Completions 错误：{chat_error}; Responses 错误：{responses_error}"
            ) from responses_error

        if getattr(fallback, "output_text", None):
            return fallback.output_text.strip()
        raise RuntimeError("图片识别失败：模型在 Responses 接口返回空内容。") from chat_error
