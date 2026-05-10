from __future__ import annotations

import base64
import os
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from auto_report_agent.settings import initialize_runtime


initialize_runtime()


def analyze_image_content(
    image_bytes: bytes,
    mime_type: str = "image/png",
    instruction: str = "",
) -> str:
    """Use the configured OpenAI-compatible model to analyze an uploaded image.

    Returns a Markdown text containing OCR result, visual description, key facts,
    and suggested next-step task interpretation.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_API_BASE", "").strip().rstrip("/")
    model = os.getenv("OPENAI_VISION_MODEL_NAME", os.getenv("OPENAI_MODEL_NAME", "gpt-5.5")).strip()

    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY，无法进行图片识别。")
    if not base_url:
        raise RuntimeError("缺少 OPENAI_API_BASE，无法进行图片识别。")
    if not image_bytes:
        raise RuntimeError("图片内容为空。")

    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt = f"""
你是一个严谨的多模态信息分析助手。请识别用户上传图片中的内容，并输出结构化 Markdown。

请完成：
1. OCR：尽可能完整提取图片中的文字；
2. 图像理解：说明图片类型、主体、界面/图表/文档结构；
3. 关键信息：列出可用于后续研究或写作的事实、数字、实体、任务要求；
4. 后续任务建议：根据图片内容判断用户可能希望系统完成什么工作；
5. 如果图片信息不足或不清晰，请明确说明。

用户补充指令：{instruction or '无'}
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
            " 请检查 OPENAI_API_KEY、OPENAI_API_BASE 是否匹配。"
        ) from exc
    except (APIConnectionError, APITimeoutError) as exc:
        raise RuntimeError(
            f"图片识别失败：网络连接异常（{exc.__class__.__name__}）。"
            " 请检查网络或反向代理。"
        ) from exc
    except RateLimitError as exc:
        raise RuntimeError(
            "图片识别失败：触发限流（429）。请稍后重试或更换模型/账号。"
        ) from exc
    except (BadRequestError, NotFoundError) as chat_error:
        # Chat Completions 拒绝请求或路径不存在，多半是兼容商把视觉接口放在 Responses API 上。
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
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
            raise RuntimeError(
                f"图片识别失败：网络连接异常（{exc.__class__.__name__}）。"
            ) from exc
        except RateLimitError as exc:
            raise RuntimeError("图片识别失败：触发限流（429）。") from exc
        except (BadRequestError, NotFoundError) as responses_error:
            raise RuntimeError(
                "图片识别失败：当前模型或兼容接口可能不支持视觉输入。"
                f" Chat Completions 错误：{chat_error}; Responses 错误：{responses_error}"
            ) from responses_error

        if getattr(response, "output_text", None):
            return response.output_text.strip()
        raise RuntimeError("图片识别失败：模型在 Responses 接口返回空内容。")

    raise RuntimeError("图片识别失败：模型没有返回可用文本。")