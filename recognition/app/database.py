from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recognition.app.preprocessing import normalize_embedding
from recognition.app.utils import utc_timestamp

SCHEMA_VERSION = 2


class LegacyDatabaseError(RuntimeError):
    """Raised when a name-keyed version-1 gallery is opened without migration."""


@dataclass(frozen=True)
class ImageEnrollmentRecord:
    image_path: str
    status: str
    note: str
    face_count: int
    used_for_embedding: bool


@dataclass(frozen=True)
class IdentityRecord:
    person_id: str
    display_name: str
    class_name: str | None
    embedding: np.ndarray
    embedding_index: int
    usable_images: int
    created_at: str
    updated_at: str
    source_images: list[str]

    @property
    def name(self) -> str:
        """Compatibility alias for presentation-only callers."""
        return self.display_name


class FaceEmbeddingsDatabase:
    """SQLite gallery keyed by stable person ID with multi-template storage."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            legacy_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(persons)").fetchall()
            }
            if legacy_columns and "person_id" not in legacy_columns:
                raise LegacyDatabaseError(
                    "This is a version-1 name-keyed gallery. Run "
                    "`python -m recognition.scripts.migrate_gallery --source <old.db> "
                    "--destination <new.db>`; the database was not modified."
                )

            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    class_name TEXT,
                    usable_images INTEGER NOT NULL CHECK (usable_images >= 0),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_row_id INTEGER NOT NULL,
                    embedding_index INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL CHECK (embedding_dim > 0),
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (person_row_id) REFERENCES persons(id) ON DELETE CASCADE,
                    UNIQUE (person_row_id, embedding_index)
                );

                CREATE TABLE IF NOT EXISTS source_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_row_id INTEGER NOT NULL,
                    image_path TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('used', 'skipped')),
                    note TEXT NOT NULL,
                    face_count INTEGER NOT NULL CHECK (face_count >= 0),
                    used_for_embedding INTEGER NOT NULL CHECK (used_for_embedding IN (0, 1)),
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (person_row_id) REFERENCES persons(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS ix_persons_display_name
                    ON persons(display_name COLLATE NOCASE);
                CREATE INDEX IF NOT EXISTS ix_source_images_person
                    ON source_images(person_row_id);
                """
            )
            row = connection.execute("SELECT version FROM schema_info LIMIT 1").fetchone()
            if row is None:
                connection.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
            elif int(row["version"]) != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported gallery schema version {row['version']}; expected {SCHEMA_VERSION}."
                )

    @staticmethod
    def _embedding_to_blob(embedding: np.ndarray) -> bytes:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("Embedding must contain finite values.")
        return vector.tobytes()

    @staticmethod
    def _blob_to_embedding(blob: bytes, dim: int) -> np.ndarray:
        vector = np.frombuffer(blob, dtype=np.float32)
        if vector.size != dim:
            raise ValueError(f"Embedding size mismatch: expected {dim}, found {vector.size}.")
        return vector.copy()

    def upsert_person(
        self,
        person_id: str,
        display_name: str,
        embeddings: list[np.ndarray],
        usable_images: int,
        image_records: list[ImageEnrollmentRecord],
        class_name: str | None = None,
    ) -> None:
        stable_id = person_id.strip()
        name = " ".join(display_name.split())
        if not stable_id:
            raise ValueError("person_id is required")
        if not name:
            raise ValueError("display_name is required")
        if not embeddings:
            raise ValueError("At least one representative embedding is required")

        timestamp = utc_timestamp()
        vectors = [normalize_embedding(item) for item in embeddings]
        dimensions = {vector.size for vector in vectors}
        if len(dimensions) != 1:
            raise ValueError("All embeddings for a person must have the same dimension")

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id, created_at FROM persons WHERE person_id = ? COLLATE NOCASE",
                (stable_id,),
            ).fetchone()
            if existing is None:
                cursor = connection.execute(
                    """
                    INSERT INTO persons (
                        person_id, display_name, class_name, usable_images, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (stable_id, name, class_name, usable_images, timestamp, timestamp),
                )
                if cursor.lastrowid is None:
                    raise RuntimeError("SQLite did not return an inserted person row ID")
                person_row_id = int(cursor.lastrowid)
            else:
                person_row_id = int(existing["id"])
                connection.execute(
                    """
                    UPDATE persons
                    SET display_name = ?, class_name = ?, usable_images = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (name, class_name, usable_images, timestamp, person_row_id),
                )
                connection.execute(
                    "DELETE FROM face_embeddings WHERE person_row_id = ?", (person_row_id,)
                )
                connection.execute(
                    "DELETE FROM source_images WHERE person_row_id = ?", (person_row_id,)
                )

            connection.executemany(
                """
                INSERT INTO face_embeddings (
                    person_row_id, embedding_index, embedding, embedding_dim, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        person_row_id,
                        index,
                        self._embedding_to_blob(vector),
                        int(vector.size),
                        timestamp,
                    )
                    for index, vector in enumerate(vectors)
                ],
            )
            if image_records:
                connection.executemany(
                    """
                    INSERT INTO source_images (
                        person_row_id, image_path, status, note, face_count,
                        used_for_embedding, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            person_row_id,
                            record.image_path,
                            record.status,
                            record.note,
                            record.face_count,
                            int(record.used_for_embedding),
                            timestamp,
                        )
                        for record in image_records
                    ],
                )

    def load_identities(self) -> list[IdentityRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.id, p.person_id, p.display_name, p.class_name, p.usable_images,
                       p.created_at, p.updated_at, e.embedding_index, e.embedding,
                       e.embedding_dim
                FROM persons AS p
                JOIN face_embeddings AS e ON e.person_row_id = p.id
                ORDER BY p.person_id, e.embedding_index
                """
            ).fetchall()
            image_rows = connection.execute(
                "SELECT person_row_id, image_path FROM source_images ORDER BY id"
            ).fetchall()

        images: dict[int, list[str]] = {}
        for row in image_rows:
            images.setdefault(int(row["person_row_id"]), []).append(str(row["image_path"]))

        return [
            IdentityRecord(
                person_id=str(row["person_id"]),
                display_name=str(row["display_name"]),
                class_name=str(row["class_name"]) if row["class_name"] is not None else None,
                embedding=self._blob_to_embedding(row["embedding"], int(row["embedding_dim"])),
                embedding_index=int(row["embedding_index"]),
                usable_images=int(row["usable_images"]),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
                source_images=images.get(int(row["id"]), []),
            )
            for row in rows
        ]

    def count_identities(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM persons").fetchone()
        return int(row["count"])
