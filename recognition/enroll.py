from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

from recognition import config
from recognition.app.database import FaceEmbeddingsDatabase, ImageEnrollmentRecord
from recognition.app.dataset_scanner import (
    DatasetPersonEntry,
    scan_known_faces_dataset,
    select_dataset_entries,
)
from recognition.app.face_engine import FaceEngine
from recognition.app.preprocessing import average_normalized_embeddings, select_largest_face
from recognition.app.utils import ensure_runtime_directories, read_image, setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersonEnrollmentSummary:
    person_id: str
    display_name: str
    class_name: str | None
    dataset_path: str
    total_images: int
    used_images: int
    skipped_images: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enroll known people from local folders of images."
    )
    parser.add_argument(
        "--person",
        help="Optional single person folder name under data/known_faces/ to enroll.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def enroll_person(
    person_entry: DatasetPersonEntry,
    engine: FaceEngine,
    database: FaceEmbeddingsDatabase,
) -> PersonEnrollmentSummary | None:
    person_name = person_entry.display_name
    image_paths = person_entry.image_paths

    if not image_paths:
        LOGGER.warning(
            "No supported images found for '%s' in %s",
            person_name,
            person_entry.person_dir,
        )
        return None

    image_records: list[ImageEnrollmentRecord] = []
    embeddings = []
    skipped_images = 0

    if person_entry.class_name:
        LOGGER.info(
            "Enrolling '%s' from class '%s' using %d images",
            person_name,
            person_entry.class_name,
            len(image_paths),
        )
    else:
        LOGGER.info("Enrolling '%s' from %d images", person_name, len(image_paths))

    for image_path in image_paths:
        image = read_image(image_path)
        if image is None:
            skipped_images += 1
            image_records.append(
                ImageEnrollmentRecord(
                    image_path=str(image_path),
                    status="skipped",
                    note="unreadable_image",
                    face_count=0,
                    used_for_embedding=False,
                )
            )
            continue

        try:
            faces = engine.detect_faces(image)
        except Exception as exc:
            LOGGER.warning("Failed to analyze %s: %s", image_path, exc)
            skipped_images += 1
            image_records.append(
                ImageEnrollmentRecord(
                    image_path=str(image_path),
                    status="skipped",
                    note=f"analysis_error:{exc}",
                    face_count=0,
                    used_for_embedding=False,
                )
            )
            continue

        if not faces:
            LOGGER.warning("No faces found in %s", image_path)
            skipped_images += 1
            image_records.append(
                ImageEnrollmentRecord(
                    image_path=str(image_path),
                    status="skipped",
                    note="no_face_detected",
                    face_count=0,
                    used_for_embedding=False,
                )
            )
            continue

        selected_face = select_largest_face(faces)
        assert selected_face is not None

        if len(faces) > 1:
            LOGGER.warning(
                "Multiple faces found in %s. Using the largest face for enrollment.",
                image_path,
            )
            note = "multiple_faces_used_largest"
        else:
            note = "single_face_used"

        try:
            embedding = engine.extract_normalized_embedding(selected_face)
        except Exception as exc:
            LOGGER.warning("Failed to extract embedding from %s: %s", image_path, exc)
            skipped_images += 1
            image_records.append(
                ImageEnrollmentRecord(
                    image_path=str(image_path),
                    status="skipped",
                    note=f"embedding_error:{exc}",
                    face_count=len(faces),
                    used_for_embedding=False,
                )
            )
            continue

        embeddings.append(embedding)
        image_records.append(
            ImageEnrollmentRecord(
                image_path=str(image_path),
                status="used",
                note=note,
                face_count=len(faces),
                used_for_embedding=True,
            )
        )

    if not embeddings:
        LOGGER.error(
            "No usable face images remained for '%s'. Existing database entry was left unchanged.",
            person_name,
        )
        return PersonEnrollmentSummary(
            person_id=person_entry.person_id,
            display_name=person_name,
            class_name=person_entry.class_name,
            dataset_path=person_entry.relative_path,
            total_images=len(image_paths),
            used_images=0,
            skipped_images=skipped_images,
        )

    representative_embedding = average_normalized_embeddings(embeddings)
    database.upsert_person(
        person_id=person_entry.person_id,
        display_name=person_name,
        class_name=person_entry.class_name,
        embeddings=[representative_embedding],
        usable_images=len(embeddings),
        image_records=image_records,
    )

    LOGGER.info(
        "Saved '%s' with %d usable images and %d skipped images",
        person_name,
        len(embeddings),
        skipped_images,
    )

    return PersonEnrollmentSummary(
        person_id=person_entry.person_id,
        display_name=person_name,
        class_name=person_entry.class_name,
        dataset_path=person_entry.relative_path,
        total_images=len(image_paths),
        used_images=len(embeddings),
        skipped_images=skipped_images,
    )


def run_enrollment(person_name: str | None = None, verbose: bool = False) -> int:
    setup_logging("DEBUG" if verbose else config.LOG_LEVEL)
    ensure_runtime_directories()

    if not config.KNOWN_FACES_DIR.exists():
        LOGGER.error("Known faces directory is missing: %s", config.KNOWN_FACES_DIR)
        return 1

    person_entries = scan_known_faces_dataset(config.KNOWN_FACES_DIR)
    if not person_entries:
        LOGGER.error(
            "No supported person folders found under %s. Use either "
            "data/known_faces/<person_name>/ or "
            "data/known_faces/<class_name>/<person_name>/",
            config.KNOWN_FACES_DIR,
        )
        return 1

    if person_name:
        try:
            person_entries = select_dataset_entries(person_entries, person_name)
        except ValueError as exc:
            LOGGER.error("%s", exc)
            return 1
        if not person_entries:
            LOGGER.error(
                "Person folder not found for selector '%s' under %s",
                person_name,
                config.KNOWN_FACES_DIR,
            )
            return 1

    database = FaceEmbeddingsDatabase(config.DATABASE_PATH)
    engine = FaceEngine()

    summaries: list[PersonEnrollmentSummary] = []
    for person_entry in person_entries:
        summary = enroll_person(person_entry, engine, database)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        LOGGER.error("Enrollment did not process any identities.")
        return 1

    enrolled_people = sum(1 for summary in summaries if summary.used_images > 0)
    total_used = sum(summary.used_images for summary in summaries)
    total_skipped = sum(summary.skipped_images for summary in summaries)

    LOGGER.info(
        "Enrollment complete. People updated: %d | Images used: %d | Images skipped: %d | Database: %s",
        enrolled_people,
        total_used,
        total_skipped,
        config.DATABASE_PATH,
    )
    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_enrollment(person_name=args.person, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
