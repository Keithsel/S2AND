import json
import logging
import os
import tempfile
from hashlib import sha256
from urllib.parse import urlparse

import requests

from s2and.consts import CACHE_ROOT

logger = logging.getLogger("s2and")

ARTIFACTS_CACHE = str(CACHE_ROOT / "artifacts")
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_DOWNLOAD_TIMEOUT_SECONDS = 60


def cached_path(url_or_filename: str | os.PathLike[str], cache_dir: str | None = None) -> str:
    """Resolve a local file path or download a URL into the artifact cache."""
    source = os.fspath(url_or_filename)
    parsed = urlparse(source)

    if parsed.scheme in {"http", "https"}:
        return get_from_cache(source, cache_dir)
    if os.path.exists(source):
        return source
    if parsed.scheme:
        raise ValueError(f"unable to parse {source} as a URL or as a local path")
    raise FileNotFoundError(f"file {source} not found")


def url_to_filename(url: str, etag: str | None = None) -> str:
    """Convert a URL and optional remote validator into a deterministic cache filename."""
    basename = os.path.basename(urlparse(url).path) or "download"
    filename = sha256(url.encode("utf-8")).hexdigest()
    if etag:
        filename += "." + sha256(etag.encode("utf-8")).hexdigest()
    return f"{filename}.{basename}"


def get_from_cache(url: str, cache_dir: str | None = None) -> str:
    """Download a URL into the artifact cache if it is not already present."""
    if cache_dir is None:
        cache_dir = ARTIFACTS_CACHE

    os.makedirs(cache_dir, exist_ok=True)
    remote_validator = _remote_cache_validator(url)
    cache_path = os.path.join(cache_dir, url_to_filename(url, remote_validator))
    if os.path.exists(cache_path):
        return cache_path
    legacy_cache_path = _legacy_cache_path(url, remote_validator, cache_dir)
    if legacy_cache_path is not None and os.path.exists(legacy_cache_path):
        return legacy_cache_path

    logger.info("%s not found in cache; downloading to %s", url, cache_path)
    _download_to_cache(url, cache_path)
    _write_metadata(cache_path, {"url": url, "etag": remote_validator})
    return cache_path


def _legacy_cache_path(url: str, remote_validator: str | None, cache_dir: str) -> str | None:
    """Return the pre-0.50 cache path for validator-prefixed artifact names."""
    if remote_validator is None:
        return None
    for prefix in ("etag:", "last-modified:"):
        if remote_validator.startswith(prefix):
            return os.path.join(cache_dir, url_to_filename(url, remote_validator.removeprefix(prefix)))
    return None


def _remote_cache_validator(url: str) -> str | None:
    response = requests.head(url, allow_redirects=True, timeout=_DOWNLOAD_TIMEOUT_SECONDS)
    try:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise OSError(f"HEAD request failed for url {url}") from exc
        etag = response.headers.get("ETag")
        if etag:
            return f"etag:{etag}"
        last_modified = response.headers.get("Last-Modified")
        if last_modified:
            return f"last-modified:{last_modified}"
        return None
    finally:
        response.close()


def _download_to_cache(url: str, cache_path: str) -> None:
    cache_dir = os.path.dirname(cache_path)
    temp_fd, temp_path = tempfile.mkstemp(
        dir=cache_dir,
        prefix=f".{os.path.basename(cache_path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(temp_fd, "wb") as temp_file:
            response = requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT_SECONDS)
            try:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        temp_file.write(chunk)
            finally:
                response.close()
        os.replace(temp_path, cache_path)
        temp_path = ""
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def _write_metadata(cache_path: str, metadata: dict[str, str | None]) -> None:
    metadata_path = f"{cache_path}.json"
    cache_dir = os.path.dirname(cache_path)
    temp_fd, temp_path = tempfile.mkstemp(
        dir=cache_dir,
        prefix=f".{os.path.basename(metadata_path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as temp_file:
            json.dump(metadata, temp_file, sort_keys=True)
        os.replace(temp_path, metadata_path)
        temp_path = ""
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
