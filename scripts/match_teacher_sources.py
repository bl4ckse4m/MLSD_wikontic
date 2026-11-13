"""Resolve teacher sample/context IDs to public HotpotQA and MuSiQue text records."""

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "kg_dump_hotpot_gpt4_1_onto_triplets.json": "hotpotqa/distractor/validation",
    "kg_dump_musique_gpt4_1_onto_triplets.json": "musique_ans_v1.0_dev",
    "kg_dump_triplets_musique_train_split_occ_openai_gpt_oss_120b_onto.json": (
        "musique_full_v1.0_train"
    ),
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def teacher_key(row: dict, source: str) -> str:
    """Keep answerability variants distinct; Full repeats base IDs across variants."""
    if source == "musique_full_v1.0_train":
        suffix = "Answerable" if row["answerable"] else "Unanswerable"
        return f"{row['id']}_{suffix}"
    return row["id"]


def source_rows(raw_dir: Path, source: str):
    if source.startswith("hotpotqa/"):
        parquet = pq.ParquetFile(raw_dir / "hotpotqa/validation.parquet")
        for batch in parquet.iter_batches(columns=["id", "context"], batch_size=256):
            for row in batch.to_pylist():
                context = row["context"]
                row["contexts"] = {
                    str(index): {"title": title, "text": " ".join(sentences).strip()}
                    for index, (title, sentences) in enumerate(
                        zip(context["title"], context["sentences"], strict=True)
                    )
                }
                yield row
    else:
        with zipfile.ZipFile(raw_dir / "musique/musique_v1.0.zip") as archive:
            with archive.open(f"data/{source}.jsonl") as stream:
                for line in stream:
                    row = json.loads(line)
                    row["contexts"] = {
                        str(paragraph["idx"]): {
                            "title": paragraph["title"],
                            "text": paragraph["paragraph_text"],
                        }
                        for paragraph in row["paragraphs"]
                    }
                    yield row


def match_dump(teacher_path: Path, raw_dir: Path, source: str, output) -> tuple[dict, set]:
    teacher = json.loads(teacher_path.read_text(encoding="utf-8"))
    matched = set()
    hashes = set()
    missing_contexts = []
    empty_contexts = []
    matched_contexts = 0
    source_count = 0
    answerability = Counter()
    for row in source_rows(raw_dir, source):
        source_count += 1
        key = teacher_key(row, source)
        if key not in teacher:
            continue
        if key in matched:
            raise ValueError(f"Ambiguous source key {key} in {source}")
        matched.add(key)
        if "answerable" in row:
            answerability[str(row["answerable"]).lower()] += 1
        for context_id in teacher[key]:
            context = row["contexts"].get(context_id)
            if context is None:
                missing_contexts.append([key, context_id])
                continue
            text = context["text"]
            if not text.strip():
                empty_contexts.append([key, context_id])
                continue
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            hashes.add(text_hash)
            matched_contexts += 1
            output.write(
                json.dumps(
                    {
                        "teacher_file": teacher_path.name,
                        "teacher_sample_id": key,
                        "source": source,
                        "source_sample_id": row["id"],
                        "answerable": row.get("answerable"),
                        "context_id": context_id,
                        "title": context["title"],
                        "text": text,
                        "text_sha256": text_hash,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    summary = {
        "teacher_file": teacher_path.name,
        "teacher_sha256": digest(teacher_path),
        "source": source,
        "source_records": source_count,
        "teacher_samples": len(teacher),
        "matched_samples": len(matched),
        "unmatched_sample_ids": sorted(set(teacher) - matched),
        "teacher_contexts": sum(len(contexts) for contexts in teacher.values()),
        "matched_contexts": matched_contexts,
        "missing_contexts": missing_contexts,
        "empty_contexts": empty_contexts,
        "unique_texts_exact": len(hashes),
        "answerability": dict(answerability),
    }
    return summary, hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", type=Path, default=ROOT / "data/teacher")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/source_matching.json")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    text_sets = {}
    with (args.output_dir / "matched_contexts.jsonl").open("w", encoding="utf-8") as output:
        for name, source in SOURCES.items():
            summary, hashes = match_dump(args.teacher_dir / name, args.raw_dir, source, output)
            summaries.append(summary)
            text_sets[name] = hashes
            print(json.dumps(summary, ensure_ascii=False))
    names = list(text_sets)
    report = {
        "method": "Exact sample ID plus source context index; not semantic fact validation",
        "source_manifest": json.loads(
            (args.raw_dir / "source_manifest.json").read_text(encoding="utf-8")
        ),
        "dumps": summaries,
        "exact_text_overlap": [
            {
                "left": left,
                "right": right,
                "unique_shared_texts": len(text_sets[left] & text_sets[right]),
            }
            for index, left in enumerate(names)
            for right in names[index + 1 :]
        ],
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {args.report}")
    if any(
        s["unmatched_sample_ids"] or s["missing_contexts"] or s["empty_contexts"] for s in summaries
    ):
        raise SystemExit("Some references did not resolve; see report")


if __name__ == "__main__":
    main()
