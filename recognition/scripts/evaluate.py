from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PairScore:
    same_person: bool
    similarity: float
    inference_ms: float | None
    end_to_end_ms: float | None


@dataclass(frozen=True)
class Metrics:
    threshold: float
    true_accepts: int
    false_accepts: int
    true_rejects: int
    false_rejects: int
    precision: float
    recall: float
    false_acceptance_rate: float
    false_rejection_rate: float


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def calculate_metrics(scores: list[PairScore], threshold: float) -> Metrics:
    true_accepts = sum(item.same_person and item.similarity >= threshold for item in scores)
    false_rejects = sum(item.same_person and item.similarity < threshold for item in scores)
    false_accepts = sum(not item.same_person and item.similarity >= threshold for item in scores)
    true_rejects = sum(not item.same_person and item.similarity < threshold for item in scores)
    return Metrics(
        threshold=threshold,
        true_accepts=true_accepts,
        false_accepts=false_accepts,
        true_rejects=true_rejects,
        false_rejects=false_rejects,
        precision=_ratio(true_accepts, true_accepts + false_accepts),
        recall=_ratio(true_accepts, true_accepts + false_rejects),
        false_acceptance_rate=_ratio(false_accepts, false_accepts + true_rejects),
        false_rejection_rate=_ratio(false_rejects, false_rejects + true_accepts),
    )


def threshold_sweep(scores: list[PairScore], thresholds: list[float]) -> list[Metrics]:
    """Calculate the same reproducible metrics for each caller-supplied threshold."""
    return [calculate_metrics(scores, threshold) for threshold in thresholds]


def _optional_float(value: str | None) -> float | None:
    return float(value) if value and value.strip() else None


def load_scores(path: Path) -> list[PairScore]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"same_person", "similarity"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires same_person and similarity columns")
        rows = [
            PairScore(
                same_person=row["same_person"].strip().casefold() in {"1", "true", "yes"},
                similarity=float(row["similarity"]),
                inference_ms=_optional_float(row.get("inference_ms")),
                end_to_end_ms=_optional_float(row.get("end_to_end_ms")),
            )
            for row in reader
        ]
    if not rows:
        raise ValueError("Evaluation CSV contains no measurements")
    return rows


def latency_summary(scores: list[PairScore]) -> dict[str, float | None]:
    inference = [item.inference_ms for item in scores if item.inference_ms is not None]
    end_to_end = [item.end_to_end_ms for item in scores if item.end_to_end_ms is not None]
    average_inference = statistics.fmean(inference) if inference else None
    return {
        "average_inference_ms": average_inference,
        "average_end_to_end_ms": statistics.fmean(end_to_end) if end_to_end else None,
        "approximate_fps": 1000.0 / average_inference if average_inference else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate consent-based pair scores.")
    parser.add_argument("scores", type=Path)
    parser.add_argument("--threshold", action="append", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scores = load_scores(args.scores)
    result = {
        "sample_count": len(scores),
        "metrics": [asdict(item) for item in threshold_sweep(scores, args.threshold)],
        "latency": latency_summary(scores),
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
