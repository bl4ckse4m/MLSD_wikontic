"""Check HW1 deliverables directly, without Git, models, or external services."""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from urllib.parse import unquote

import yaml

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "README.md",
    "pyproject.toml",
    "requirements-dev.txt",
    ".pre-commit-config.yaml",
    ".gitignore",
    ".github/workflows/checks.yml",
    "docs/ml_system_design_doc.md",
    "src/wikontic_lite/__init__.py",
    "scripts/check_hw1.py",
)
DEMO_TEXT = "Marie Curie was born in Warsaw."


def check_markdown(path: Path, text: str) -> None:
    """Validate local file links and JSON examples; do not claim Mermaid rendering."""
    fences = re.findall(r"^```.*$", text, flags=re.MULTILINE)
    if len(fences) % 2:
        raise ValueError(f"{path}: unbalanced code fences")
    for raw in re.findall(r"\]\(([^\s)]+)\)", text):
        if raw.startswith(("https://", "http://", "mailto:", "#")):
            continue
        target = unquote(raw.split("#", 1)[0])
        if not (path.parent / target).exists():
            raise ValueError(f"{path}: missing link target {target}")
    for block in re.findall(r"```json\n(.*?)\n```", text, flags=re.DOTALL):
        payload = json.loads(block)
        if payload.get("document_id") != "demo-001":
            continue
        for triple in payload["triples"]:
            evidence = triple["evidence"]
            start, end = evidence["start"], evidence["end"]
            if not (0 <= start < end <= len(DEMO_TEXT)):
                raise ValueError(f"{path}: invalid demo evidence offsets")
            if DEMO_TEXT[start:end] != evidence["text"]:
                raise ValueError(f"{path}: demo evidence does not match input")


def check_files() -> None:
    for name in FILES:
        path = ROOT / name
        data = path.read_bytes()
        text = data.decode("utf-8")
        if len(data) > 500 * 1024:
            raise ValueError(f"{name}: exceeds 500 KiB")
        if not data.endswith(b"\n") or data.endswith(b"\n\n"):
            raise ValueError(f"{name}: expected a single final newline")
        for number, line in enumerate(text.splitlines(), start=1):
            if line.rstrip(" \t") != line:
                raise ValueError(f"{name}:{number}: trailing whitespace")
            if re.match(r"^(<{7}|={7}|>{7})(?:\s|$)", line):
                raise ValueError(f"{name}:{number}: conflict marker")
        if path.suffix in {".yml", ".yaml"}:
            yaml.safe_load(text)
        elif path.suffix == ".toml":
            tomllib.loads(text)
        elif path.suffix == ".md":
            check_markdown(path, text)
        print(f"PASS {name}")

    design = (ROOT / "docs/ml_system_design_doc.md").read_text(encoding="utf-8")
    diagrams = re.findall(r"```mermaid\n(.*?)\n```", design, flags=re.DOTALL)
    if len(diagrams) != 3 or any(not d.startswith("flowchart TD\n") for d in diagrams):
        raise ValueError("Expected three top-down Mermaid blocks")
    requirements = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    ruff_version = re.search(r"^ruff==(.+)$", requirements, flags=re.MULTILINE).group(1)
    ruff_repo = next(repo for repo in config["repos"] if "ruff-pre-commit" in repo["repo"])
    if ruff_repo["rev"] != f"v{ruff_version}":
        raise ValueError("Ruff environment and hook versions differ")
    print("PASS Ruff version agreement and Mermaid block count (not rendering)")


def smoke_ruff() -> None:
    """Prove the configured linter rejects F401, then check the clean codebase."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="lint_smoke_",
        dir=ROOT / "src",
        encoding="utf-8",
    ) as handle:
        handle.write("import os\n")
        handle.flush()
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", handle.name],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 1 or "F401" not in result.stdout:
            raise ValueError(f"Negative lint smoke test failed: {result.stdout}{result.stderr}")
    subprocess.run([sys.executable, "-m", "ruff", "check", "."], cwd=ROOT, check=True)
    print("PASS negative Ruff smoke test; temporary file removed; final lint passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-ruff", action="store_true")
    args = parser.parse_args()
    check_files()
    if args.smoke_ruff:
        smoke_ruff()


if __name__ == "__main__":
    main()
