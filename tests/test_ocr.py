"""Tests for the OCR abstraction and its Tesseract -> RapidOCR fallback."""
import pytest
from PIL import Image

from src.ingestion.parsers import ocr


def _image():
    return Image.new("RGB", (60, 20), "white")


def test_unknown_engine_raises_value_error():
    with pytest.raises(ValueError, match="Unknown OCR engine"):
        ocr.ocr_image(_image(), engine="bogus")


def test_no_engine_available_raises_runtime_error(monkeypatch):
    monkeypatch.setattr(ocr, "ocr_available", lambda _name: False)
    with pytest.raises(RuntimeError, match="No OCR engine available"):
        ocr.ocr_image_with_engine(_image())


def test_primary_engine_used_when_it_produces_text(monkeypatch):
    monkeypatch.setattr(ocr, "ocr_available", lambda _name: True)
    monkeypatch.setattr(ocr, "_run", lambda name, _img: f"text from {name}")

    text, engine = ocr.ocr_image_with_engine(_image(), engine="tesseract")

    assert engine == "tesseract"
    assert text == "text from tesseract"


def test_falls_back_when_primary_returns_empty(monkeypatch):
    monkeypatch.setattr(ocr, "ocr_available", lambda _name: True)

    def fake_run(name, _img):
        return "" if name == "tesseract" else "rapid text"

    monkeypatch.setattr(ocr, "_run", fake_run)

    text, engine = ocr.ocr_image_with_engine(_image(), engine="tesseract")

    assert engine == "rapidocr"
    assert text == "rapid text"


def test_falls_back_when_primary_raises(monkeypatch):
    monkeypatch.setattr(ocr, "ocr_available", lambda _name: True)

    def fake_run(name, _img):
        if name == "tesseract":
            raise RuntimeError("tesseract exploded")
        return "rapid text"

    monkeypatch.setattr(ocr, "_run", fake_run)

    text, engine = ocr.ocr_image_with_engine(_image(), engine="tesseract")

    assert engine == "rapidocr"
    assert text == "rapid text"


def test_single_available_engine_is_used(monkeypatch):
    monkeypatch.setattr(ocr, "ocr_available", lambda name: name == "rapidocr")
    monkeypatch.setattr(ocr, "_run", lambda name, _img: f"{name} only")

    text, engine = ocr.ocr_image_with_engine(_image(), engine="tesseract")

    assert engine == "rapidocr"
    assert text == "rapidocr only"


def test_all_available_engines_empty_returns_none(monkeypatch):
    monkeypatch.setattr(ocr, "ocr_available", lambda _name: True)
    monkeypatch.setattr(ocr, "_run", lambda _name, _img: "   ")

    text, engine = ocr.ocr_image_with_engine(_image(), engine="tesseract")

    assert text == ""
    assert engine is None


def test_ocr_image_returns_only_text(monkeypatch):
    monkeypatch.setattr(ocr, "ocr_available", lambda _name: True)
    monkeypatch.setattr(ocr, "_run", lambda _name, _img: "hello")

    assert ocr.ocr_image(_image()) == "hello"


def test_extract_rapidocr_lines_handles_v3_object():
    class Output:
        txts = ("line one", "", "line two")

    assert ocr._extract_rapidocr_lines(Output()) == ["line one", "line two"]


def test_extract_rapidocr_lines_handles_v2_rows():
    rows = [
        ([[0, 0], [1, 1]], "first", 0.9),
        ([[0, 0], [1, 1]], "second", 0.8),
    ]
    assert ocr._extract_rapidocr_lines((rows, 0.01)) == ["first", "second"]
