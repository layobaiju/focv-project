"""Download logic tested offline by replacing the network calls."""

import hashlib
import io

import pytest

from visionact import download


def entry(path: str, payload: bytes = b"video-bytes", kind: str = "file") -> dict:
    return {"type": kind, "path": path, "size": len(payload), "lfs": {"oid": hashlib.sha256(payload).hexdigest()}}


def test_listing_keeps_only_well_formed_clips_of_the_class(monkeypatch):
    listing = {
        "train/PushUps": [
            entry("train/PushUps/v_PushUps_g08_c01.avi"),
            entry("train/PushUps/v_PullUps_g08_c01.avi"),  # wrong class in the name
            entry("train/PushUps/../../evil.avi"),  # path traversal attempt
            entry("train/PushUps/notes.txt"),
            entry("train/PushUps/subfolder", kind="directory"),
        ],
        "test/PushUps": [entry("test/PushUps/v_PushUps_g01_c01.avi")],
    }
    monkeypatch.setattr(download, "_list_directory", lambda repo, path: listing[path])
    files = download.list_class_files("PushUps")
    assert [f.name for f in files] == ["v_PushUps_g08_c01.avi", "v_PushUps_g01_c01.avi"]
    assert all(f.class_name == "PushUps" and f.sha256 for f in files)


def test_unknown_class_gives_a_helpful_error(monkeypatch):
    monkeypatch.setattr(download, "_list_directory", lambda repo, path: [])
    with pytest.raises(ValueError, match="case-sensitive"):
        download.list_class_files("Sitting")


def test_download_verifies_checksum_and_skips_existing_files(monkeypatch, tmp_path):
    payload = b"fake avi content"
    remote = download.RemoteFile("test/PushUps/v_PushUps_g01_c01.avi", "PushUps", "v_PushUps_g01_c01.avi", len(payload), hashlib.sha256(payload).hexdigest())
    calls = []

    def fake_open(url, timeout=60.0):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(download, "_open", fake_open)
    assert download.download_file(remote, tmp_path) == "downloaded"
    target = tmp_path / "PushUps" / "v_PushUps_g01_c01.avi"
    assert target.read_bytes() == payload
    assert download.download_file(remote, tmp_path) == "skipped" and len(calls) == 1

    # A corrupted local copy is detected and replaced.
    target.write_bytes(b"x" * len(payload))
    assert download.download_file(remote, tmp_path) == "downloaded"
    assert target.read_bytes() == payload


def test_download_rejects_content_that_fails_the_checksum(monkeypatch, tmp_path):
    remote = download.RemoteFile("test/PushUps/v_PushUps_g01_c01.avi", "PushUps", "v_PushUps_g01_c01.avi", 4, "0" * 64)
    monkeypatch.setattr(download, "_open", lambda url, timeout=60.0: io.BytesIO(b"abcd"))
    monkeypatch.setattr(download.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="Could not download"):
        download.download_file(remote, tmp_path, retries=2)
    assert not list((tmp_path / "PushUps").glob("*")), "no partial or unverified file may be left behind"


def test_declining_the_prompt_downloads_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "_list_directory", lambda repo, path: [entry(f"{path}/v_PushUps_g01_c01.avi")])
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    counts = download.download_subset(("PushUps",), tmp_path)
    assert counts == {"downloaded": 0, "skipped": 0, "failed": 0}
    assert not any(tmp_path.iterdir())
