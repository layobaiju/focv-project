"""Download a few UCF101 classes instead of the full 6.9 GB archive.

The official distribution (https://www.crcv.ucf.edu/data/UCF101.php) is a single
RAR file. The Hugging Face dataset ``sumith2425/UCF101`` is a third-party copy
that stores every clip as its own file, so only the classes named in the config
have to be fetched. Each file is checked against the size and SHA-256 published
by the host, and file names must follow the UCF101 pattern.

Clips are written in the official layout, ``<dataset_dir>/<Class>/v_<Class>_gXX_cYY.avi``.
The copy's own train/test folders are ignored: VisionAct derives its split from
the group number in the file name (see :mod:`visionact.dataset`).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .utils import get_logger

log = get_logger(__name__)

HF_REPO = "sumith2425/UCF101"
_TREE_URL = "https://huggingface.co/api/datasets/{repo}/tree/main/{path}?limit=1000"
_FILE_URL = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
_REMOTE_SPLITS = ("train", "test")
_USER_AGENT = "visionact/0.1 (academic project)"
_NAME_PATTERN = re.compile(r"^v_(?P<cls>[A-Za-z]+)_g\d{2}_c\d{2}\.avi$")
_NEXT_LINK = re.compile(r'<([^>]+)>;\s*rel="next"')


@dataclass(frozen=True)
class RemoteFile:
    remote_path: str  # e.g. "test/PushUps/v_PushUps_g01_c01.avi"
    class_name: str
    name: str
    size: int
    sha256: str | None


def _open(url: str, timeout: float = 60.0):
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout)


def _list_directory(repo: str, path: str) -> list[dict]:
    url: str | None = _TREE_URL.format(repo=repo, path=urllib.parse.quote(path))
    entries: list[dict] = []
    while url:
        with _open(url) as response:
            entries.extend(json.load(response))
            match = _NEXT_LINK.search(response.headers.get("Link", ""))
        url = match.group(1) if match else None
    return entries


def list_class_files(class_name: str, repo: str = HF_REPO) -> list[RemoteFile]:
    """List the clips of one class, rejecting anything not named like a UCF101 clip."""
    files: list[RemoteFile] = []
    for split in _REMOTE_SPLITS:
        try:
            entries = _list_directory(repo, f"{split}/{class_name}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            raise
        for entry in entries:
            name = entry["path"].rsplit("/", 1)[-1]
            match = _NAME_PATTERN.match(name)
            if entry.get("type") != "file" or not match or match.group("cls") != class_name:
                log.warning("Skipping unexpected remote entry: %s", entry.get("path"))
                continue
            sha256 = (entry.get("lfs") or {}).get("oid")
            files.append(RemoteFile(entry["path"], class_name, name, int(entry["size"]), sha256))
    if not files:
        raise ValueError(
            f"Class {class_name!r} was not found in {repo}. UCF101 class names are case-sensitive "
            "CamelCase, for example 'JumpingJack' or 'PushUps'."
        )
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_complete(path: Path, remote: RemoteFile) -> bool:
    if not path.is_file() or path.stat().st_size != remote.size:
        return False
    return remote.sha256 is None or _sha256(path) == remote.sha256


def download_file(remote: RemoteFile, dataset_dir: Path, repo: str = HF_REPO, retries: int = 4) -> str:
    """Fetch one clip. Returns "skipped" if a verified copy already exists, else "downloaded"."""
    destination = dataset_dir / remote.class_name / remote.name
    if _is_complete(destination, remote):
        return "skipped"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.parent / (destination.name + ".part")
    url = _FILE_URL.format(repo=repo, path=urllib.parse.quote(remote.remote_path))
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with _open(url, timeout=120.0) as response, partial.open("wb") as handle:
                for chunk in iter(lambda: response.read(1 << 16), b""):
                    handle.write(chunk)
            if not _is_complete(partial, remote):
                raise OSError(f"size or checksum mismatch for {remote.name}")
            partial.replace(destination)
            return "downloaded"
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            time.sleep(2.0 * (attempt + 1))
    partial.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {remote.remote_path} after {retries} attempts: {last_error}")


def plan_download(classes: tuple[str, ...], repo: str = HF_REPO) -> list[RemoteFile]:
    files: list[RemoteFile] = []
    for class_name in classes:
        class_files = list_class_files(class_name, repo)
        log.info("%-18s %4d clips, %6.1f MB", class_name, len(class_files), sum(f.size for f in class_files) / 1e6)
        files.extend(class_files)
    return files


def download_subset(
    classes: tuple[str, ...],
    dataset_dir: Path,
    assume_yes: bool = False,
    workers: int = 8,
    repo: str = HF_REPO,
) -> dict[str, int]:
    """Download the given classes after showing the size and asking for confirmation."""
    files = plan_download(classes, repo)
    total_mb = sum(f.size for f in files) / 1e6
    print(f"About to download {len(files)} clips ({total_mb:.0f} MB) from https://huggingface.co/datasets/{repo}")
    print(f"Destination: {dataset_dir}")
    if not assume_yes and input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Download cancelled.")
        return {"downloaded": 0, "skipped": 0, "failed": 0}

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_file, remote, dataset_dir, repo): remote for remote in files}
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                counts[future.result()] += 1
            except RuntimeError as exc:
                counts["failed"] += 1
                log.error("%s", exc)
            if done % 50 == 0 or done == len(files):
                log.info("%d/%d files processed", done, len(files))
    log.info("Done: %(downloaded)d downloaded, %(skipped)d already present, %(failed)d failed", counts)
    if counts["failed"]:
        log.error("Some files failed. Re-run the command; completed files are skipped.")
    return counts
