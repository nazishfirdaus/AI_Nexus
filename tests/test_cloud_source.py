"""Tests for the S3-compatible cloud source (fake client; no network)."""
import io

import pytest

from src.ingestion.sources import cloud


class FakePaginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, Bucket=None, Prefix=""):
        return iter(self._pages)


class FakeS3:
    """Minimal stand-in for a boto3 S3 client."""

    def __init__(self, pages=None, bodies=None):
        self._pages = pages or []
        self._bodies = bodies or {}
        self.get_object_calls = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakePaginator(self._pages)

    def get_object(self, Bucket=None, Key=None, Range=None):
        self.get_object_calls.append((Bucket, Key, Range))
        if Key not in self._bodies:
            raise RuntimeError("NoSuchKey")
        data = self._bodies[Key]
        if Range:
            start, end = Range.replace("bytes=", "").split("-")
            data = data[: int(end) + 1]
        return {"Body": io.BytesIO(data)}


def test_build_client_requires_keys():
    with pytest.raises(cloud.CloudError):
        cloud.build_s3_client({})


def test_build_client_uses_injected_session():
    class Session:
        def __init__(self):
            self.kwargs = None

        def client(self, service, **kwargs):
            self.kwargs = (service, kwargs)
            return "client"

    session = Session()
    built = cloud.build_s3_client(
        {
            "s3_access_key_id": "AK",
            "s3_secret_access_key": "SK",
            "s3_region": "us-east-1",
            "s3_endpoint_url": "https://minio.local",
        },
        session=session,
    )
    assert built == "client"
    service, kwargs = session.kwargs
    assert service == "s3"
    assert kwargs["region_name"] == "us-east-1"
    assert kwargs["endpoint_url"] == "https://minio.local"


def test_list_objects_filters_unsupported_and_folders():
    pages = [
        {
            "Contents": [
                {"Key": "docs/", "Size": 0},
                {"Key": "docs/loan.pdf", "Size": 100},
                {"Key": "docs/notes.txt", "Size": 200},
                {"Key": "docs/image.exe", "Size": 300},
            ]
        }
    ]
    objs = cloud.list_objects(FakeS3(pages=pages), "bucket", "docs/")
    assert [o["filename"] for o in objs] == ["loan.pdf", "notes.txt"]


def test_list_objects_respects_max_files():
    pages = [{"Contents": [{"Key": f"f{i}.pdf", "Size": 10} for i in range(10)]}]
    objs = cloud.list_objects(FakeS3(pages=pages), "bucket", max_files=3)
    assert len(objs) == 3


def test_list_objects_skips_oversized():
    pages = [{"Contents": [{"Key": "big.pdf", "Size": 5000}, {"Key": "small.pdf", "Size": 10}]}]
    objs = cloud.list_objects(FakeS3(pages=pages), "bucket", max_bytes=100)
    assert [o["filename"] for o in objs] == ["small.pdf"]


def test_get_object_bytes_returns_content_and_caps():
    client = FakeS3(bodies={"a.txt": b"0123456789"})
    assert cloud.get_object_bytes(client, "b", "a.txt") == b"0123456789"
    assert cloud.get_object_bytes(client, "b", "a.txt", max_bytes=4) == b"0123"


def test_get_object_bytes_wraps_errors():
    client = FakeS3(bodies={})
    with pytest.raises(cloud.CloudError):
        cloud.get_object_bytes(client, "b", "missing.txt")


def test_collect_supported_objects_builds_items():
    pages = [{"Contents": [{"Key": "docs/loan.pdf", "Size": 4}, {"Key": "docs/x.exe", "Size": 4}]}]
    client = FakeS3(pages=pages, bodies={"docs/loan.pdf": b"data"})
    items = cloud.collect_supported_objects(client, "bucket", "docs/")
    assert items == [(b"data", "loan.pdf")]
