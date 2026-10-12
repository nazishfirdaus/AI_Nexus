"""OCR abstraction with a Tesseract -> RapidOCR fallback.

The baseline engine is Tesseract (``pytesseract``). When it is unavailable or
returns no text, RapidOCR is tried if installed. Both engines are imported
lazily, so this module costs nothing when the optional extras are absent and
the baseline keeps working exactly as before.

Selection
---------
``engine`` (or ``settings.OCR_ENGINE``) names the *primary* engine; the other
engine is the automatic fallback. If only one engine is installed it is used;
if neither is installed a clear error is raised.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from config import settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image

TESSERACT = "tesseract"
RAPIDOCR = "rapidocr"
ENGINES = (TESSERACT, RAPIDOCR)

_rapidocr_engine = None  # cached RapidOCR() instance (model load is expensive)


def _normalize(engine: str | None) -> str:
    name = (engine or settings.OCR_ENGINE or TESSERACT).strip().lower()
    if name not in ENGINES:
        raise ValueError(
            f"Unknown OCR engine '{name}'. Choose from: {', '.join(ENGINES)}"
        )
    return name


def _fallback_order(primary: str) -> list[str]:
    return [primary, *(e for e in ENGINES if e != primary)]


def _tesseract_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _rapidocr_available() -> bool:
    try:
        import rapidocr  # noqa: F401
    except Exception:
        return False
    return True


def ocr_available(engine: str) -> bool:
    """Return True if the named engine can be used in this environment."""

    name = _normalize(engine)
    return _tesseract_available() if name == TESSERACT else _rapidocr_available()


def _run_tesseract(image: "Image") -> str:
    import pytesseract

    return pytesseract.image_to_string(image)


def _extract_rapidocr_lines(result) -> list[str]:
    """Return recognised text lines from either the v3 or v2 RapidOCR result."""

    if result is None:
        return []

    txts = getattr(result, "txts", None)

    # v2 returns (rows, elapsed); rows are [box, text, score] triples.
    if txts is None and isinstance(result, (tuple, list)) and len(result) == 2:
        first = result[0]
        txts = getattr(first, "txts", None)
        if txts is None:
            result = first

    if txts is None:
        if isinstance(result, (list, tuple)):
            return [
                item[1]
                for item in result
                if isinstance(item, (list, tuple))
                and len(item) >= 2
                and isinstance(item[1], str)
            ]
        return []

    return [line for line in txts if line]


def _run_rapidocr(image: "Image") -> str:
    global _rapidocr_engine

    import numpy as np
    from rapidocr import RapidOCR

    if _rapidocr_engine is None:
        _rapidocr_engine = RapidOCR()

    array = np.asarray(image.convert("RGB"))
    result = _rapidocr_engine(array)
    return "\n".join(_extract_rapidocr_lines(result))


def _run(engine: str, image: "Image") -> str:
    return _run_tesseract(image) if engine == TESSERACT else _run_rapidocr(image)


def ocr_image_with_engine(
    image: "Image",
    *,
    engine: str | None = None,
) -> tuple[str, str | None]:
    """OCR ``image``, returning ``(text, engine_used)``.

    ``engine_used`` is None when no engine produced text. Raises RuntimeError
    when no OCR engine is installed at all.
    """

    order = _fallback_order(_normalize(engine))
    available = [name for name in order if ocr_available(name)]

    if not available:
        raise RuntimeError(
            "No OCR engine available. Install Tesseract (baseline) or run "
            "'pip install -r requirements-ocr.txt' for the RapidOCR fallback."
        )

    for name in available:
        try:
            text = _run(name, image)
        except Exception:
            text = ""
        if text and text.strip():
            return text, name

    return "", None


def ocr_image(image: "Image", *, engine: str | None = None) -> str:
    """OCR ``image`` using the primary engine with automatic fallback."""

    text, _ = ocr_image_with_engine(image, engine=engine)
    return text
