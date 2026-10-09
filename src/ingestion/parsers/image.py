"""Image parsing: raster files go straight to Tesseract OCR.

An image is a single page (page_number=1) flagged with ocr_used=True so the
UI/registry stats show that extraction was optical, not textual.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from config.settings import IMAGE_EXTENSIONS, OCR_DPI

from .common import make_document, normalize_text, validate_file

ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS


def parse_image(
    file_path: str | Path,
    dpi: int = OCR_DPI,
) -> list[Document]:
    """OCR an image into a single page-level Document."""

    path = validate_file(file_path, ALLOWED_EXTENSIONS)

    from PIL import Image, ImageOps

    import pytesseract

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("L")

        zoom = max(1.0, dpi / 96)
        if zoom > 1.0:
            image = image.resize(
                (int(image.width * zoom), int(image.height * zoom)),
                Image.LANCZOS,
            )

        extracted_text = pytesseract.image_to_string(image)

    normalized = normalize_text(extracted_text)

    if not normalized:
        return []

    return [
        make_document(
            normalized,
            filename=path.name,
            page_number=1,
            ocr_used=True,
        )
    ]
