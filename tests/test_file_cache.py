import json
from pathlib import Path

import pytest

import s2and.file_cache as file_cache


class _HeadResponse:
    def __init__(self, headers: dict[str, str] | None = None, status_code: int = 200):
        self.headers = headers or {}
        self.status_code = status_code
        self.closed = False

    def raise_for_status(self):
        if self.status_code != 200:
            raise file_cache.requests.HTTPError(f"HEAD failed with status {self.status_code}")

    def close(self):
        self.closed = True


class _GetResponse:
    def __init__(self, chunks: list[bytes], status_code: int = 200):
        self.chunks = chunks
        self.status_code = status_code
        self.closed = False
        self.chunk_size = None

    def raise_for_status(self):
        if self.status_code != 200:
            raise file_cache.requests.HTTPError(f"GET failed with status {self.status_code}")

    def iter_content(self, chunk_size: int):
        self.chunk_size = chunk_size
        yield from self.chunks

    def close(self):
        self.closed = True


def test_get_from_cache_download_does_not_write_stdout(monkeypatch, tmp_path, capsys):
    url = "https://example.org/model.bin"
    etag = "etag-123"
    expected_bytes = b"cached model payload"
    head_response = _HeadResponse(headers={"ETag": etag})
    response = _GetResponse([b"cached model ", b"", b"payload"])

    def _fake_head(request_url: str, *, allow_redirects: bool = False, timeout: int | None = None):
        assert request_url == url
        assert allow_redirects is True
        assert timeout == file_cache._DOWNLOAD_TIMEOUT_SECONDS
        return head_response

    def _fake_get(request_url: str, *, stream: bool = False, timeout: int | None = None):
        assert request_url == url
        assert stream is True
        assert timeout == file_cache._DOWNLOAD_TIMEOUT_SECONDS
        return response

    monkeypatch.setattr(file_cache.requests, "head", _fake_head)
    monkeypatch.setattr(file_cache.requests, "get", _fake_get)

    cache_path = file_cache.get_from_cache(url, str(tmp_path))

    captured = capsys.readouterr()
    assert captured.out == ""

    assert Path(cache_path).read_bytes() == expected_bytes
    assert Path(cache_path).name == file_cache.url_to_filename(url, f"etag:{etag}")
    assert json.loads(Path(cache_path + ".json").read_text(encoding="utf-8")) == {
        "url": url,
        "etag": f"etag:{etag}",
    }
    assert head_response.closed is True
    assert response.closed is True
    assert response.chunk_size == file_cache._DOWNLOAD_CHUNK_SIZE


def test_get_from_cache_hit_checks_remote_validator_but_does_not_download(monkeypatch, tmp_path):
    url = "https://example.org/model.bin"
    etag = "etag-123"
    cache_path = tmp_path / file_cache.url_to_filename(url, f"etag:{etag}")
    cache_path.write_bytes(b"already cached")
    head_response = _HeadResponse(headers={"ETag": etag})

    def _fake_head(request_url: str, *, allow_redirects: bool = False, timeout: int | None = None):
        assert request_url == url
        assert allow_redirects is True
        assert timeout == file_cache._DOWNLOAD_TIMEOUT_SECONDS
        return head_response

    def _unexpected_get(*args, **kwargs):
        raise AssertionError("cached artifact should not be downloaded again")

    monkeypatch.setattr(file_cache.requests, "head", _fake_head)
    monkeypatch.setattr(file_cache.requests, "get", _unexpected_get)

    assert file_cache.get_from_cache(url, str(tmp_path)) == str(cache_path)
    assert head_response.closed is True


def test_get_from_cache_ignores_raw_etag_filename(monkeypatch, tmp_path):
    url = "https://example.org/model.bin"
    etag = "etag-123"
    raw_validator_cache_path = tmp_path / file_cache.url_to_filename(url, etag)
    raw_validator_cache_path.write_bytes(b"raw validator cached")
    head_response = _HeadResponse(headers={"ETag": etag})
    get_response = _GetResponse([b"validator namespaced"])

    monkeypatch.setattr(
        file_cache.requests,
        "head",
        lambda request_url, *, allow_redirects=False, timeout=None: head_response,
    )
    monkeypatch.setattr(
        file_cache.requests,
        "get",
        lambda request_url, *, stream=False, timeout=None: get_response,
    )

    cache_path = file_cache.get_from_cache(url, str(tmp_path))

    assert Path(cache_path).name == file_cache.url_to_filename(url, f"etag:{etag}")
    assert Path(cache_path).read_bytes() == b"validator namespaced"
    assert raw_validator_cache_path.read_bytes() == b"raw validator cached"
    assert head_response.closed is True


def test_get_from_cache_downloads_new_file_when_etag_changes(monkeypatch, tmp_path):
    url = "https://example.org/model.bin"
    old_path = tmp_path / file_cache.url_to_filename(url, "etag:old")
    old_path.write_bytes(b"old")
    head_response = _HeadResponse(headers={"ETag": "new"})
    get_response = _GetResponse([b"new"])

    monkeypatch.setattr(
        file_cache.requests,
        "head",
        lambda request_url, *, allow_redirects=False, timeout=None: head_response,
    )
    monkeypatch.setattr(file_cache.requests, "get", lambda request_url, *, stream=False, timeout=None: get_response)

    cache_path = file_cache.get_from_cache(url, str(tmp_path))

    assert Path(cache_path).name == file_cache.url_to_filename(url, "etag:new")
    assert Path(cache_path).read_bytes() == b"new"
    assert old_path.read_bytes() == b"old"


def test_get_from_cache_uses_last_modified_validator(monkeypatch, tmp_path):
    url = "https://example.org/model.bin"
    last_modified = "Wed, 21 Oct 2015 07:28:00 GMT"
    head_response = _HeadResponse(headers={"Last-Modified": last_modified})
    get_response = _GetResponse([b"new"])

    monkeypatch.setattr(
        file_cache.requests,
        "head",
        lambda request_url, *, allow_redirects=False, timeout=None: head_response,
    )
    monkeypatch.setattr(file_cache.requests, "get", lambda request_url, *, stream=False, timeout=None: get_response)

    cache_path = file_cache.get_from_cache(url, str(tmp_path))

    assert Path(cache_path).name == file_cache.url_to_filename(url, f"last-modified:{last_modified}")
    assert Path(cache_path).read_bytes() == b"new"


def test_get_from_cache_head_failure_preserves_oserror_contract(monkeypatch, tmp_path):
    url = "https://example.org/model.bin"
    head_response = _HeadResponse(status_code=503)

    monkeypatch.setattr(
        file_cache.requests,
        "head",
        lambda request_url, *, allow_redirects=False, timeout=None: head_response,
    )

    with pytest.raises(OSError, match="HEAD request failed"):
        file_cache.get_from_cache(url, str(tmp_path))
    assert head_response.closed is True


def test_download_to_cache_removes_temp_file_on_failure(monkeypatch, tmp_path):
    url = "https://example.org/model.bin"
    target_path = tmp_path / file_cache.url_to_filename(url, "etag:bad")
    get_response = _GetResponse([b"ignored"], status_code=500)

    monkeypatch.setattr(file_cache.requests, "get", lambda request_url, *, stream=False, timeout=None: get_response)

    with pytest.raises(file_cache.requests.HTTPError):
        file_cache._download_to_cache(url, str(target_path))

    assert not target_path.exists()
    assert list(tmp_path.iterdir()) == []
    assert get_response.closed is True


def test_cached_path_returns_existing_local_file(tmp_path):
    local_path = tmp_path / "artifact.bin"
    local_path.write_bytes(b"local")

    assert file_cache.cached_path(local_path) == str(local_path)
