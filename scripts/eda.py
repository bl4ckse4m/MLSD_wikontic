"""Reproduce descriptive EDA of aligned HotpotQA/MuSiQue teacher data on CPU."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "H": "kg_dump_hotpot_gpt4_1_onto_triplets.json",
    "M": "kg_dump_musique_gpt4_1_onto_triplets.json",
    "G": "kg_dump_triplets_musique_train_split_occ_openai_gpt_oss_120b_onto.json",
}
LABELS = {"H": "HotpotQA / GPT", "M": "MuSiQue dev / GPT", "G": "MuSiQue train / OSS"}
FIELDS = ("initial_triplets", "triplets", "ontology_filtered_triplets", "filtered_triplets")
REQUIRED = ("subject", "relation", "object")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm(text: str) -> str:
    return " ".join(text.split())


def spo(triple: dict) -> tuple | None:
    if not isinstance(triple, dict):
        return None
    if not all(isinstance(triple.get(key), str) and triple[key].strip() for key in REQUIRED):
        return None
    return tuple(norm(triple[key]) for key in REQUIRED)


def stats(values: list) -> dict:
    if not values:
        return {"n": 0}
    array = np.asarray(values)
    return {
        "n": len(values),
        "min": int(array.min()),
        "mean": round(float(array.mean()), 3),
        "p50": round(float(np.percentile(array, 50)), 3),
        "p90": round(float(np.percentile(array, 90)), 3),
        "p95": round(float(np.percentile(array, 95)), 3),
        "p99": round(float(np.percentile(array, 99)), 3),
        "max": int(array.max()),
    }


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inspect_triples(triples, text: str) -> tuple[Counter, set, Counter, Counter]:
    """Count all list elements; invalid_spo includes non-dicts and invalid SPO fields.

    Type, qualifier, evidence and lexical counters apply only to valid SPO records.
    A malformed container is counted separately, not interpreted as zero facts.
    """
    quality = Counter()
    relations, types = Counter(), Counter()
    if not isinstance(triples, list):
        quality["triplets_not_list"] += 1
        return quality, set(), relations, types
    full, spos = [], []
    for triple in triples:
        quality["triples"] += 1
        if not isinstance(triple, dict):
            quality["malformed_triple"] += 1
            quality["invalid_spo"] += 1
            continue
        if any(key not in triple for key in REQUIRED):
            quality["missing_required"] += 1
        if any(key in triple and not isinstance(triple[key], str) for key in REQUIRED):
            quality["invalid_required_type"] += 1
        if any(isinstance(triple.get(key), str) and not triple[key].strip() for key in REQUIRED):
            quality["empty_required"] += 1
        item = spo(triple)
        if item is None:
            quality["invalid_spo"] += 1
            continue
        full.append(json.dumps(triple, ensure_ascii=False, sort_keys=True))
        spos.append(item)
        relations[item[1]] += 1
        quality["self_link"] += item[0].casefold() == item[2].casefold()
        quality["subject_not_literal"] += item[0].casefold() not in text.casefold()
        quality["object_not_literal"] += item[2].casefold() not in text.casefold()
        for key in ("subject_type", "object_type"):
            if isinstance(triple.get(key), str) and triple[key].strip():
                types[triple[key].strip()] += 1
            else:
                quality["missing_or_invalid_type"] += 1
        qualifiers = triple.get("qualifiers")
        if not isinstance(qualifiers, list):
            quality["invalid_qualifiers"] += 1
        else:
            quality["with_qualifiers"] += bool(qualifiers)
            quality["qualifiers"] += len(qualifiers)
            for qualifier in qualifiers:
                if not isinstance(qualifier, dict) or not all(
                    isinstance(qualifier.get(key), str) and qualifier[key].strip()
                    for key in ("relation", "object")
                ):
                    quality["invalid_qualifier_item"] += 1
        quality["with_evidence_field"] += "evidence" in triple
    quality["duplicate_full"] += len(full) - len(set(full))
    quality["duplicate_spo"] += len(spos) - len(set(spos))
    return quality, set(spos), relations, types


class Groups:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, first: int, second: int) -> None:
        first, second = self.find(first), self.find(second)
        self.parent[max(first, second)] = min(first, second)


def group_records(rows: list[dict], path: Path, seed: int) -> dict:
    if not rows:
        raise ValueError("Cannot split an empty dataset")
    groups = Groups(len(rows))
    text_index, title_index = {}, {}
    for index, row in enumerate(rows):
        key = row["normalized_hash"]
        if key in text_index:
            groups.union(index, text_index[key])
        text_index.setdefault(key, index)
        title = norm(row["title"]).casefold()
        if title:
            if title in title_index:
                groups.union(index, title_index[title])
            title_index.setdefault(title, index)
    members = defaultdict(list)
    for index in range(len(rows)):
        members[groups.find(index)].append(index)
    stable = {
        root: text_hash("\n".join(sorted(rows[i]["record_id"] for i in indices)))
        for root, indices in members.items()
    }
    order = sorted(stable.values())
    if len(order) < 2:
        raise ValueError("Train/validation requires at least two independent groups")
    random.Random(seed).shuffle(order)
    validation_size = min(len(order) - 1, max(1, round(len(order) * 0.1)))
    validation = set(order[:validation_size])
    part_texts = defaultdict(set)
    part_titles = defaultdict(set)
    counts = Counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            group = stable[groups.find(index)]
            part = "validation_candidate" if group in validation else "train_candidate"
            counts[part] += 1
            part_texts[part].add(row["normalized_hash"])
            title = norm(row["title"]).casefold()
            if title:
                part_titles[part].add(title)
            stream.write(
                json.dumps(
                    {"record_id": row["record_id"], "group_id": group, "candidate_split": part}
                )
                + "\n"
            )
    keys = ("train_candidate", "validation_candidate")
    text_overlap = len(part_texts[keys[0]] & part_texts[keys[1]])
    title_overlap = len(part_titles[keys[0]] & part_titles[keys[1]])
    if text_overlap or title_overlap:
        raise ValueError("Candidate groups leaked normalized texts or titles")
    return {
        "groups": len(members),
        "largest_group_records": max(map(len, members.values())),
        "group_size": stats(list(map(len, members.values()))),
        "candidate_counts": dict(counts),
        "normalized_text_overlap": text_overlap,
        "title_overlap": title_overlap,
        "policy": "Union equal whitespace-normalized texts or casefolded titles across sources",
        "status": "Candidate teacher train/validation only; no independent gold test",
    }


def select_diagnostics(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    selected = []
    for alias in FILES:
        pool = [row for row in rows if row["dataset"] == alias]
        if not pool:
            continue
        chosen = {}

        def add(row: dict, reason: str) -> None:
            if row["record_id"] not in chosen:
                chosen[row["record_id"]] = {"row": row, "reason": reason}

        add(max(pool, key=lambda row: row["text_chars"]), "longest_text")
        add(max(pool, key=lambda row: row["triples"]), "most_triples")
        rare = [row for row in pool if row["has_rare_relation"]]
        suspicious = [row for row in pool if row["self_links"] or row["invalid_spo"]]
        for candidates, reason in ((rare, "rare_relation"), (suspicious, "structural_signal")):
            for row in rng.sample(candidates, min(2, len(candidates))):
                add(row, reason)
        for row in rng.sample(pool, len(pool)):
            if len(chosen) >= 10:
                break
            add(row, "seeded_random")
        for item in chosen.values():
            row = item["row"]
            selected.append(
                {
                    "record_id": row["record_id"],
                    "dataset": alias,
                    "reason": item["reason"],
                    "teacher_sample_id": row["teacher_sample_id"],
                    "context_id": row["context_id"],
                    "title": row["title"],
                    "text_chars": row["text_chars"],
                    "triples": row["triples"],
                }
            )
    return selected


def diagnostic_fact(triples: list, reason: str, frequencies: dict) -> dict | None:
    """Choose one inspectable fact per selected context, without quality labels."""
    if not isinstance(triples, list):
        return None
    records = [triple for triple in triples if isinstance(triple, dict)]
    if not records:
        return None
    if reason == "structural_signal":
        return next(
            (
                triple
                for triple in records
                if spo(triple) is None or spo(triple)[0].casefold() == spo(triple)[2].casefold()
            ),
            records[0],
        )
    if reason == "rare_relation":
        return next(
            (
                t
                for t in records
                if isinstance(t.get("relation"), str)
                and t["relation"].strip()
                and frequencies.get(norm(t["relation"]), 0) <= 5
            ),
            records[0],
        )
    return next((t for t in records if t.get("qualifiers")), records[0])


def plots(rows: list[dict], summaries: dict, processed: Path, figures: Path) -> None:
    plt.rcParams.update({"svg.hashsalt": "wikontic-eda-v1", "font.size": 10})
    metrics = (
        ("text_chars", "Символов во входном тексте", "text_lengths.svg"),
        ("triples", "Триплетов на контекст", "triple_counts.svg"),
        ("target_chars", "Символов в JSON-ответе", "target_lengths.svg"),
    )
    for metric, xlabel, name in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        for ax, alias in zip(axes, FILES, strict=True):
            values = [row[metric] for row in rows if row["dataset"] == alias]
            ax.hist(values, bins=40, color="#3678a4")
            ax.set(title=LABELS[alias], xlabel=xlabel, ylabel="Контекстных записей")
            ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(figures / name, metadata={"Date": None})
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, alias in zip(axes, FILES, strict=True):
        top = summaries[alias]["top_relations"][:12]
        ax.barh([item[0] for item in top][::-1], [item[1] for item in top][::-1])
        ax.set(title=LABELS[alias], xlabel="Частота отношения")
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(figures / "relations.svg", metadata={"Date": None})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for alias in FILES:
        frequencies = sorted(summaries[alias]["relation_frequencies"].values(), reverse=True)
        ax.plot(
            range(1, len(frequencies) + 1), np.cumsum(frequencies) / sum(frequencies), label=alias
        )
    ax.set(xlabel="Отношений по убыванию частоты", ylabel="Накопленная доля триплетов")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "relation_tail.svg", metadata={"Date": None})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    aliases = list(FILES)
    ax.bar(aliases, [summaries[a]["contexts"] for a in aliases], label="Все записи")
    ax.bar(
        aliases, [summaries[a]["unique_texts_exact"] for a in aliases], label="Уникальные тексты"
    )
    ax.set(ylabel="Контекстов", xlabel="Источник")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "repeats.svg", metadata={"Date": None})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for index, field in enumerate(("initial_triplets", "triplets")):
        values = [summaries[a]["field_totals"][field]["triples"] for a in ("H", "M")]
        ax.bar(np.arange(2) + (index - 0.5) * 0.35, values, width=0.35, label=field)
    ax.set(xticks=np.arange(2), xticklabels=["H", "M"], ylabel="Триплетов")
    ax.set_title("G: initial_triplets отсутствует, сравнение недоступно")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "stages.svg", metadata={"Date": None})
    plt.close(fig)
    columns = ["dataset", "record_id", "text_chars", "text_words", "triples", "target_chars"]
    write_csv(
        processed / "context_lengths.csv", [{key: row[key] for key in columns} for row in rows]
    )
    for path in figures.glob("*.svg"):
        text = path.read_text(encoding="utf-8")
        path.write_text(
            "\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/eda")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed/eda")
    args = parser.parse_args()
    tables, figures = args.report_dir / "tables", args.report_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    source_report = json.loads((ROOT / "reports/source_matching.json").read_text())
    expected = {item["teacher_file"]: item["teacher_sha256"] for item in source_report["dumps"]}
    for source in source_report["source_manifest"]["sources"]:
        if sha(ROOT / source["path"]) != source["sha256"]:
            raise ValueError(f"Source changed: {source['path']}")
    aligned_path = ROOT / "data/processed/matched_contexts.jsonl"
    aligned = {}
    with aligned_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["teacher_file"], row["teacher_sample_id"], row["context_id"])
            if key in aligned or text_hash(row["text"]) != row["text_sha256"]:
                raise ValueError(f"Invalid or duplicate aligned context: {key}")
            aligned[key] = row
    rows, summaries = [], {}
    dumps = {}
    text_sets, normal_sets, titles = {}, {}, {}
    facts_by_text = defaultdict(lambda: defaultdict(set))
    for alias, name in FILES.items():
        path = ROOT / "data/teacher" / name
        if sha(path) != expected[name]:
            raise ValueError(f"Teacher data changed: {name}")
        data = json.loads(path.read_text(encoding="utf-8"))
        dumps[alias] = data
        quality, fields, transitions = Counter(), defaultdict(Counter), Counter()
        relations, types = Counter(), Counter()
        exact, normalized, title_set = set(), set(), set()
        local_rows = []
        for sample_id, contexts in data.items():
            for context_id, entry in contexts.items():
                source = aligned[(name, sample_id, context_id)]
                text = source["text"]
                exact.add(source["text_sha256"])
                normal_hash = text_hash(norm(text))
                normalized.add(normal_hash)
                title_set.add(norm(source["title"]).casefold())
                q, fact_set, rel, typ = inspect_triples(entry.get("triplets"), text)
                quality.update(q)
                quality["contexts_with_valid_spo"] += bool(fact_set)
                relations.update(rel)
                types.update(typ)
                for field in FIELDS:
                    if field in entry:
                        fields[field]["contexts"] += 1
                        value = entry[field]
                        fields[field]["triples"] += len(value) if isinstance(value, list) else 0
                        fields[field]["not_list"] += not isinstance(value, list)
                if "initial_triplets" in entry and isinstance(entry["initial_triplets"], list):
                    initial = {spo(t) for t in entry["initial_triplets"]} - {None}
                    transitions["compared_contexts"] += 1
                    transitions["initial_unique_spo"] += len(initial)
                    transitions["final_unique_spo"] += len(fact_set)
                    transitions["retained_exact_spo"] += len(initial & fact_set)
                    transitions["removed_or_changed_spo"] += len(initial - fact_set)
                    transitions["added_or_changed_spo"] += len(fact_set - initial)
                facts_by_text[normal_hash][alias].update(fact_set)
                triples = entry.get("triplets", [])
                count = len(triples) if isinstance(triples, list) else 0
                quality["empty_targets"] += isinstance(triples, list) and not triples
                item = {
                    "dataset": alias,
                    "record_id": f"{alias}/{sample_id}/{context_id}",
                    "teacher_sample_id": sample_id,
                    "context_id": context_id,
                    "title": source["title"],
                    "text_hash": source["text_sha256"],
                    "normalized_hash": normal_hash,
                    "text_chars": len(text),
                    "text_words": len(text.split()),
                    "triples": count,
                    "target_chars": len(json.dumps({"triplets": triples}, ensure_ascii=False)),
                    "self_links": q["self_link"],
                    "invalid_spo": q["invalid_spo"],
                    "relation_names": list(rel),
                    "answerable": source["answerable"],
                }
                local_rows.append(item)
        for item in local_rows:
            item["has_rare_relation"] = any(relations[name] <= 5 for name in item["relation_names"])
        rows.extend(local_rows)
        text_sets[alias], normal_sets[alias], titles[alias] = exact, normalized, title_set
        summaries[alias] = {
            "file": name,
            "sha256": expected[name],
            "qa_samples": len(data),
            "contexts": len(local_rows),
            "unique_texts_exact": len(exact),
            "unique_texts_whitespace_normalized": len(normalized),
            "unique_titles_casefolded": len(title_set),
            "answerability_contexts": dict(Counter(str(item["answerable"]) for item in local_rows)),
            "quality": dict(quality),
            "field_totals": {field: dict(counts) for field, counts in fields.items()},
            "initial_final_spo": dict(transitions),
            "unique_relations": len(relations),
            "unique_types": len(types),
            "top10_relation_share": sum(count for _, count in relations.most_common(10))
            / sum(relations.values()),
            "relations_frequency_le_5": sum(count <= 5 for count in relations.values()),
            "top_relations": relations.most_common(20),
            "relation_frequencies": dict(sorted(relations.items())),
            "type_frequencies": dict(sorted(types.items())),
            "lengths": {
                metric: stats([item[metric] for item in local_rows])
                for metric in ("text_chars", "text_words", "triples", "target_chars")
            },
        }
        print(
            alias,
            json.dumps({key: summaries[alias][key] for key in ("contexts", "quality", "lengths")}),
            flush=True,
        )
    if len(rows) != len(aligned):
        raise ValueError("Some aligned records were not consumed")
    overlaps = []
    for index, left in enumerate(FILES):
        for right in list(FILES)[index + 1 :]:
            shared = normal_sets[left] & normal_sets[right]
            similarities = []
            for key in sorted(shared):
                first, second = facts_by_text[key][left], facts_by_text[key][right]
                if first | second:
                    similarities.append(len(first & second) / len(first | second))
            overlaps.append(
                {
                    "left": left,
                    "right": right,
                    "exact_texts": len(text_sets[left] & text_sets[right]),
                    "normalized_texts": len(shared),
                    "shared_titles": len(titles[left] & titles[right]),
                    "spo_jaccard_pairs": len(similarities),
                    "spo_jaccard_mean": float(np.mean(similarities)) if similarities else None,
                    "spo_sets_differ": sum(value != 1 for value in similarities),
                }
            )
    groups = group_records(rows, args.processed_dir / "group_manifest.jsonl", args.seed)
    diagnostics = select_diagnostics(rows, args.seed)
    write_json(tables / "diagnostic_sample.json", diagnostics)
    write_csv(tables / "diagnostic_sample.csv", diagnostics)
    selected_facts = []
    with (args.processed_dir / "diagnostic_contexts.jsonl").open("w", encoding="utf-8") as stream:
        for item in diagnostics:
            alias, sample_id, context_id = item["record_id"].split("/")
            source = aligned[(FILES[alias], sample_id, context_id)]
            entry = dumps[alias][sample_id][context_id]
            fact = diagnostic_fact(
                entry["triplets"], item["reason"], summaries[alias]["relation_frequencies"]
            )
            selected_facts.append({**item, "selected_fact": fact})
            stream.write(
                json.dumps(
                    {**item, "text": source["text"], "target": dumps[alias][sample_id][context_id]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    write_json(tables / "diagnostic_facts.json", selected_facts)
    summary = {
        "seed": args.seed,
        "versions": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ("numpy", "matplotlib", "pyarrow")
            },
        },
        "aligned_sha256": sha(aligned_path),
        "sources": source_report["source_manifest"]["sources"],
        "datasets": summaries,
        "overlaps": overlaps,
        "grouping": groups,
        "diagnostic_contexts": len(diagnostics),
        "limitations": [
            "No semantic gold labels",
            "No tokenizer counts",
            "No near-duplicate search",
            "No final test or training",
        ],
    }
    write_json(tables / "summary.json", summary)
    write_csv(tables / "overlaps.csv", overlaps)
    relation_rows = [
        {"dataset": alias, "relation": name, "count": count}
        for alias in FILES
        for name, count in sorted(summaries[alias]["relation_frequencies"].items())
    ]
    write_csv(tables / "relations.csv", relation_rows)
    plots(rows, summaries, args.processed_dir, figures)
    print("Groups:", json.dumps(groups), flush=True)
    print("Overlaps:", json.dumps(overlaps), flush=True)
    print(f"Saved {args.report_dir}; diagnostic texts and candidate manifest: {args.processed_dir}")


if __name__ == "__main__":
    main()
