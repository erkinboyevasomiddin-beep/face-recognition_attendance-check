from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

import numpy as np

from recognition.app.database import FaceEmbeddingsDatabase, ImageEnrollmentRecord


def _load_mapping(path: Path | None) -> dict[str, tuple[str, str]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"legacy_name", "person_id", "display_name"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Mapping CSV requires legacy_name, person_id, display_name")
        return {row["legacy_name"]: (row["person_id"], row["display_name"]) for row in reader}


def migrate(source: Path, destination: Path, mapping_path: Path | None = None) -> None:
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    mapping = _load_mapping(mapping_path)
    source_connection = sqlite3.connect(f"file:{source.resolve().as_posix()}?mode=ro", uri=True)
    source_connection.row_factory = sqlite3.Row
    destination_db = FaceEmbeddingsDatabase(destination)
    try:
        people = source_connection.execute("SELECT * FROM persons ORDER BY id").fetchall()
        for person in people:
            legacy_name = str(person["name"])
            person_id, display_name = mapping.get(
                legacy_name, (f"legacy-person-{int(person['id']):06d}", legacy_name)
            )
            image_rows = source_connection.execute(
                "SELECT * FROM source_images WHERE person_id = ? ORDER BY id",
                (person["id"],),
            ).fetchall()
            vector = np.frombuffer(person["embedding"], dtype=np.float32).copy()
            if vector.size != int(person["embedding_dim"]):
                raise ValueError(f"Corrupt embedding for legacy person row {person['id']}")
            destination_db.upsert_person(
                person_id=person_id,
                display_name=display_name,
                # sqlite3.Row intentionally has no dict.get method.
                class_name=(
                    person["class_name"] if "class_name" in person else None  # noqa: SIM401
                ),
                embeddings=[vector],
                usable_images=int(person["usable_images"]),
                image_records=[
                    ImageEnrollmentRecord(
                        image_path=str(row["image_path"]),
                        status=str(row["status"]),
                        note=str(row["note"]),
                        face_count=int(row["face_count"]),
                        used_for_embedding=bool(row["used_for_embedding"]),
                    )
                    for row in image_rows
                ],
            )
    finally:
        source_connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy a version-1 face gallery to schema v2.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--mapping", type=Path)
    args = parser.parse_args()
    migrate(args.source, args.destination, args.mapping)
    print(f"Migrated gallery to {args.destination}; the source was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
