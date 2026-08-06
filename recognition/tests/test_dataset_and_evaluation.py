from __future__ import annotations

import json

import pytest

from recognition.app.dataset_scanner import scan_known_faces_dataset, select_dataset_entries
from recognition.scripts.evaluate import (
    PairScore,
    calculate_metrics,
    latency_summary,
    threshold_sweep,
)


def test_dataset_allows_duplicate_names_but_rejects_duplicate_ids(tmp_path):
    for folder, person_id in (("first", "DEMO-1"), ("second", "DEMO-2")):
        directory = tmp_path / folder
        directory.mkdir()
        (directory / "face.jpg").write_bytes(b"placeholder")
        (directory / "person.json").write_text(
            json.dumps({"person_id": person_id, "display_name": "Same Demo Name"}),
            encoding="utf-8",
        )
    entries = scan_known_faces_dataset(tmp_path)
    assert [entry.person_id for entry in entries] == ["DEMO-1", "DEMO-2"]

    (tmp_path / "second" / "person.json").write_text(
        json.dumps({"person_id": "DEMO-1", "display_name": "Same Demo Name"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate person_id"):
        scan_known_faces_dataset(tmp_path)


def test_evaluation_metrics_and_latency_are_computed_not_fabricated():
    scores = [
        PairScore(True, 0.9, 10.0, 15.0),
        PairScore(True, 0.4, 20.0, 25.0),
        PairScore(False, 0.8, 10.0, 15.0),
        PairScore(False, 0.2, 20.0, 25.0),
    ]
    metrics = calculate_metrics(scores, 0.5)
    assert (metrics.true_accepts, metrics.false_rejects) == (1, 1)
    assert (metrics.false_accepts, metrics.true_rejects) == (1, 1)
    assert metrics.precision == 0.5 and metrics.recall == 0.5
    latency = latency_summary(scores)
    assert latency["average_inference_ms"] == 15.0
    assert latency["approximate_fps"] == pytest.approx(1000 / 15)
    sweep = threshold_sweep(scores, [0.2, 0.5, 0.9])
    assert [item.threshold for item in sweep] == [0.2, 0.5, 0.9]


def test_dataset_nested_legacy_selection_and_invalid_metadata(tmp_path):
    assert scan_known_faces_dataset(tmp_path / "missing") == []
    class_dir = tmp_path / "5-01"
    first = class_dir / "first-person"
    second = class_dir / "second-person"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "face.png").write_bytes(b"placeholder")
    (second / "face.jpeg").write_bytes(b"placeholder")
    (first / "person.json").write_text(
        json.dumps(
            {
                "person_id": "DEMO-001",
                "display_name": "Demo One",
                "class_name": "6-02",
            }
        ),
        encoding="utf-8",
    )
    entries = scan_known_faces_dataset(tmp_path)
    assert len(entries) == 2
    assert entries[0].relative_path == "6-02/first-person"
    assert entries[1].uses_legacy_identity is True
    assert select_dataset_entries(entries, "DEMO-001") == [entries[0]]
    assert select_dataset_entries(entries, "second-person") == [entries[1]]
    assert select_dataset_entries(entries, "") == []
    assert select_dataset_entries(entries, "missing") == []

    (first / "person.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        scan_known_faces_dataset(tmp_path)
    (first / "person.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Unable to read"):
        scan_known_faces_dataset(tmp_path)


@pytest.mark.parametrize(
    "metadata",
    [
        {"person_id": "bad id", "display_name": "Demo"},
        {"person_id": "DEMO-001", "display_name": ""},
    ],
)
def test_dataset_metadata_validation(tmp_path, metadata):
    person = tmp_path / "person"
    person.mkdir()
    (person / "face.jpg").write_bytes(b"placeholder")
    (person / "person.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        scan_known_faces_dataset(tmp_path)


def test_dataset_ambiguous_path_selector(tmp_path):
    entries = []
    for class_name, person_id in (("5-01", "DEMO-1"), ("6-01", "DEMO-2")):
        person = tmp_path / class_name / "same-folder"
        person.mkdir(parents=True)
        (person / "face.jpg").write_bytes(b"placeholder")
        (person / "person.json").write_text(
            json.dumps({"person_id": person_id, "display_name": "Same Demo Name"}),
            encoding="utf-8",
        )
    entries = scan_known_faces_dataset(tmp_path)
    with pytest.raises(ValueError, match="ambiguous"):
        select_dataset_entries(entries, "same-folder")
