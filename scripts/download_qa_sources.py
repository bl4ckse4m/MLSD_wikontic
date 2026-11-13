"""Download public QA sources used to resolve local teacher dump references."""

import argparse
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
import zipfile
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUSIQUE_URL = "https://drive.google.com/uc?export=download&id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"
HOTPOT_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "distractor/validation-00000-of-00001.parquet"
)
HOTPOT_SHA256 = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
MUSIQUE_SHA256 = "98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd"


class DriveForm(HTMLParser):
    """Read the public large-file confirmation form, without executing page code."""

    def __init__(self) -> None:
        super().__init__()
        self.action = ""
        self.parameters = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.action = attrs.get("action", "")
        if tag == "input" and attrs.get("type") == "hidden" and attrs.get("name"):
            self.parameters[attrs["name"]] = attrs.get("value", "")


def fingerprint(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    name = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    return {"path": str(name), "bytes": path.stat().st_size, "sha256": digest}


def download(url: str, path: Path) -> None:
    if path.exists():
        print(f"Using existing {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=180) as response:
        if "text/html" in response.headers.get("Content-Type", ""):
            form = DriveForm()
            form.feed(response.read().decode("utf-8"))
            if form.action != "https://drive.usercontent.google.com/download":
                raise ValueError(f"Unexpected HTML response from {url}")
            confirmed_url = form.action + "?" + urllib.parse.urlencode(form.parameters)
            download(confirmed_url, path)
            return
        temporary = path.with_suffix(path.suffix + ".part")
        with temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        temporary.rename(path)
    print(f"Downloaded {path}: {path.stat().st_size} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    args = parser.parse_args()
    musique = args.raw_dir / "musique/musique_v1.0.zip"
    hotpot = args.raw_dir / "hotpotqa/validation.parquet"
    download(MUSIQUE_URL, musique)
    download(HOTPOT_URL, hotpot)
    if not zipfile.is_zipfile(musique):
        raise ValueError("MuSiQue download is not a ZIP archive")
    musique_info = fingerprint(musique)
    if musique_info["sha256"] != MUSIQUE_SHA256:
        raise ValueError("MuSiQue checksum differs from the recorded v1.0 archive")
    hotpot_info = fingerprint(hotpot)
    if hotpot_info["sha256"] != HOTPOT_SHA256:
        raise ValueError("HotpotQA checksum differs from the inspected HF file metadata")
    manifest = {
        "verified_at": datetime.now(UTC).isoformat(),
        "sources": [
            {
                **musique_info,
                "url": MUSIQUE_URL,
                "homepage": "https://github.com/stonybrooknlp/musique",
                "license": "CC-BY-4.0",
                "distribution": "Official v1.0 archive linked by download_data.sh",
            },
            {
                **hotpot_info,
                "url": HOTPOT_URL,
                "homepage": "https://hotpotqa.github.io/",
                "license": "CC-BY-SA-4.0",
                "distribution": "HF hotpotqa/hotpot_qa, distractor validation, Parquet",
                "note": "Original curtis.ml.cmu.edu host did not resolve during download",
            },
        ],
    }
    path = args.raw_dir / "source_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
