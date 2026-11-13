"""Regression tests for EDA counting and grouping policies; no source corpus required."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "eda", Path(__file__).resolve().parents[1] / "scripts/eda.py"
)
eda = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eda)


def fact(**changes):
    return {
        "subject": "Alice",
        "relation": "works at",
        "object": "Lab",
        "subject_type": "human",
        "object_type": "organization",
        "qualifiers": [],
        **changes,
    }


class CountingTests(unittest.TestCase):
    def test_invalid_spo_and_literal_flags_are_separate(self):
        quality, facts, _, _ = eda.inspect_triples(
            [fact(), fact(object=""), fact(subject="A. Smith")], "Alice works at Lab."
        )
        self.assertEqual(quality["triples"], 3)
        self.assertEqual(quality["invalid_spo"], 1)
        self.assertEqual(quality["subject_not_literal"], 1)
        self.assertEqual(len(facts), 2)

    def test_qualifier_variants_are_not_full_duplicates(self):
        triples = [fact(), fact(qualifiers=[{"relation": "point in time", "object": "2020"}])]
        quality, _, _, _ = eda.inspect_triples(triples, "Alice works at Lab.")
        self.assertEqual(quality["duplicate_full"], 0)
        self.assertEqual(quality["duplicate_spo"], 1)
        self.assertEqual(quality["with_qualifiers"], 1)

    def test_identical_facts_count_as_duplicates(self):
        quality, _, _, _ = eda.inspect_triples([fact(), fact()], "Alice works at Lab.")
        self.assertEqual(quality["duplicate_full"], 1)
        self.assertEqual(quality["duplicate_spo"], 1)

    def test_non_list_and_empty_statistics(self):
        quality, _, _, _ = eda.inspect_triples(None, "Text")
        self.assertEqual(quality["triplets_not_list"], 1)
        self.assertEqual(eda.stats([]), {"n": 0})

    def test_normalization_only_changes_whitespace(self):
        self.assertEqual(eda.norm("A  B\nC"), "A B C")
        self.assertNotEqual(eda.norm("Alice"), eda.norm("alice"))

    def test_non_dict_elements_are_invalid_spo(self):
        quality, facts, _, _ = eda.inspect_triples(
            [None, "not a fact", ["Alice", "works at", "Lab"], fact()], "Alice works at Lab."
        )
        self.assertEqual(quality["triples"], 4)
        self.assertEqual(quality["malformed_triple"], 3)
        self.assertEqual(quality["invalid_spo"], 3)
        self.assertEqual(len(facts), 1)

    def test_missing_and_wrong_type_spo_fields(self):
        missing = fact()
        del missing["relation"]
        quality, facts, _, _ = eda.inspect_triples(
            [missing, fact(subject=7), fact(object=None), fact(relation="  ")], "Text"
        )
        self.assertEqual(quality["invalid_spo"], 4)
        self.assertEqual(quality["missing_required"], 1)
        self.assertEqual(quality["invalid_required_type"], 2)
        self.assertEqual(quality["empty_required"], 1)
        self.assertEqual(facts, set())

    def test_qualifier_errors_do_not_change_spo_validity(self):
        triples = [
            fact(qualifiers=None),
            fact(qualifiers=[None, {"relation": "year"}, {"relation": 5, "object": "2020"}]),
        ]
        quality, facts, _, _ = eda.inspect_triples(triples, "Alice works at Lab.")
        self.assertEqual(quality["invalid_spo"], 0)
        self.assertEqual(quality["invalid_qualifiers"], 1)
        self.assertEqual(quality["invalid_qualifier_item"], 3)
        self.assertEqual(len(facts), 1)


class DiagnosticTests(unittest.TestCase):
    def test_empty_and_malformed_inputs(self):
        self.assertEqual(eda.select_diagnostics([], seed=42), [])
        for value in (None, {}, [], [None, "broken"]):
            self.assertIsNone(eda.diagnostic_fact(value, "seeded_random", {}))

    def test_structural_selection_handles_non_string_fields(self):
        broken = fact(subject=7)
        self.assertEqual(
            eda.diagnostic_fact([None, broken, fact()], "structural_signal", {}), broken
        )

    def test_rare_selection_skips_malformed_relations(self):
        rare = fact(relation="mentors")
        self.assertEqual(
            eda.diagnostic_fact([None, fact(relation=3), rare], "rare_relation", {"mentors": 1}),
            rare,
        )


class GroupTests(unittest.TestCase):
    def test_empty_dataset_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            with self.assertRaisesRegex(ValueError, "empty dataset"):
                eda.group_records([], path, seed=42)
            self.assertFalse(path.exists())

    def test_single_group_cannot_form_train_and_validation(self):
        rows = [
            {"record_id": "H/1/0", "normalized_hash": "a", "title": "First"},
            {"record_id": "M/2/0", "normalized_hash": "a", "title": "Other"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            with self.assertRaisesRegex(ValueError, "at least two independent groups"):
                eda.group_records(rows, path, seed=42)
            self.assertFalse(path.exists())

    def test_transitive_title_and_text_grouping_is_reproducible(self):
        rows = [
            {"record_id": "H/1/0", "normalized_hash": "a", "title": "First"},
            {"record_id": "M/2/0", "normalized_hash": "b", "title": "first"},
            {"record_id": "G/3/0", "normalized_hash": "b", "title": "Other"},
            {"record_id": "G/4/0", "normalized_hash": "c", "title": "Unique"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            result = eda.group_records(rows, path, seed=42)
            original = path.read_bytes()
            eda.group_records(rows, path, seed=42)
            self.assertEqual(path.read_bytes(), original)
            records = [json.loads(line) for line in original.decode().splitlines()]
        self.assertEqual(result["groups"], 2)
        self.assertEqual(result["largest_group_records"], 3)
        self.assertEqual(result["normalized_text_overlap"], 0)
        self.assertEqual(result["title_overlap"], 0)
        self.assertEqual(len({record["group_id"] for record in records[:3]}), 1)


if __name__ == "__main__":
    unittest.main()
