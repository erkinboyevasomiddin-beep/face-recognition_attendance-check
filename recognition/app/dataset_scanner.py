from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from recognition.app.utils import is_image_file, iter_image_paths

LOGGER = logging.getLogger(__name__)
METADATA_FILENAME = "person.json"
LEGACY_NAMESPACE = uuid.UUID("7f75d875-576e-4acc-bff7-e22461e98661")
PERSON_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{1,63}$")


@dataclass(frozen=True)
class DatasetPersonEntry:
    person_id: str
    display_name: str
    person_dir: Path
    image_paths: list[Path]
    class_name: str | None = None
    uses_legacy_identity: bool = False

    @property
    def person_name(self) -> str:
        return self.display_name

    @property
    def relative_path(self) -> str:
        if self.class_name:
            return f"{self.class_name}/{self.person_dir.name}"
        return self.person_dir.name


def scan_known_faces_dataset(root_dir: Path) -> list[DatasetPersonEntry]:
    """Discover identity folders and load stable IDs from ``person.json`` metadata.

    Supported layouts remain ``<person>/images`` and ``<class>/<person>/images``.
    A metadata file is strongly recommended. Legacy folders receive a deterministic
    migration ID so the visible folder name is never sent as the system identity.
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        return []

    entries: list[DatasetPersonEntry] = []
    for top_level_dir in _visible_directories(root_dir):
        if _direct_image_paths(top_level_dir):
            entry = _build_person_entry(top_level_dir, root_dir=root_dir)
            if entry:
                entries.append(entry)
            continue

        for child_dir in _visible_directories(top_level_dir):
            entry = _build_person_entry(
                child_dir,
                root_dir=root_dir,
                class_name=top_level_dir.name,
            )
            if entry:
                entries.append(entry)

    _validate_unique_person_ids(entries)
    _log_duplicate_display_names(entries)
    return sorted(entries, key=lambda item: item.person_id.casefold())


def select_dataset_entries(
    entries: list[DatasetPersonEntry], selector: str
) -> list[DatasetPersonEntry]:
    normalized = selector.strip().replace("\\", "/").casefold()
    if not normalized:
        return []
    id_matches = [entry for entry in entries if entry.person_id.casefold() == normalized]
    if id_matches:
        return id_matches
    path_matches = [
        entry for entry in entries if entry.relative_path.casefold().endswith(normalized)
    ]
    if len(path_matches) <= 1:
        return path_matches
    raise ValueError(f"Selector '{selector}' is ambiguous; use a stable person_id.")


def _build_person_entry(
    person_dir: Path,
    *,
    root_dir: Path,
    class_name: str | None = None,
) -> DatasetPersonEntry | None:
    image_paths = iter_image_paths(person_dir)
    if not image_paths:
        return None

    metadata_path = person_dir / METADATA_FILENAME
    if metadata_path.exists():
        metadata = _read_metadata(metadata_path)
        person_id = str(metadata.get("person_id", "")).strip()
        display_name = " ".join(str(metadata.get("display_name", "")).split())
        metadata_class = metadata.get("class_name")
        resolved_class = " ".join(str(metadata_class).split()) if metadata_class else class_name
        if not PERSON_ID_PATTERN.fullmatch(person_id):
            raise ValueError(f"Invalid person_id in {metadata_path}: {person_id!r}")
        if not display_name:
            raise ValueError(f"display_name is required in {metadata_path}")
        return DatasetPersonEntry(
            person_id=person_id,
            display_name=display_name,
            person_dir=person_dir,
            image_paths=image_paths,
            class_name=resolved_class,
        )

    relative = person_dir.relative_to(root_dir).as_posix().casefold()
    legacy_id = f"legacy-{uuid.uuid5(LEGACY_NAMESPACE, relative)}"
    LOGGER.warning(
        "Legacy identity folder %s has no %s; assigned stable migration ID %s. "
        "Add metadata before renaming or moving the folder.",
        person_dir,
        METADATA_FILENAME,
        legacy_id,
    )
    return DatasetPersonEntry(
        person_id=legacy_id,
        display_name=person_dir.name,
        person_dir=person_dir,
        image_paths=image_paths,
        class_name=class_name,
        uses_legacy_identity=True,
    )


def _read_metadata(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read identity metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Identity metadata must be a JSON object: {path}")
    return value


def _direct_image_paths(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and is_image_file(path))


def _visible_directories(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_dir() and not path.name.startswith(".")
    )


def _validate_unique_person_ids(entries: list[DatasetPersonEntry]) -> None:
    seen: dict[str, Path] = {}
    for entry in entries:
        key = entry.person_id.casefold()
        if key in seen:
            raise ValueError(
                f"Duplicate person_id '{entry.person_id}' in {seen[key]} and {entry.person_dir}"
            )
        seen[key] = entry.person_dir


def _log_duplicate_display_names(entries: list[DatasetPersonEntry]) -> None:
    names: dict[str, list[str]] = {}
    for entry in entries:
        names.setdefault(entry.display_name.casefold(), []).append(entry.person_id)
    for display_name, person_ids in names.items():
        if len(person_ids) > 1:
            LOGGER.info(
                "Display name %r belongs to multiple stable IDs: %s",
                display_name,
                ", ".join(sorted(person_ids)),
            )
