import pytest

from auto_report_agent.vision import validate_image_bytes


def test_validate_image_accepts_matching_png_signature():
    validate_image_bytes(b"\x89PNG\r\n\x1a\ncontent", mime_type="image/png")


def test_validate_image_rejects_mime_mismatch():
    with pytest.raises(ValueError, match="MIME"):
        validate_image_bytes(b"not-a-png", mime_type="image/png")


def test_validate_image_rejects_oversize():
    with pytest.raises(ValueError, match="超过上限"):
        validate_image_bytes(
            b"\x89PNG\r\n\x1a\ncontent",
            mime_type="image/png",
            max_bytes=8,
        )
