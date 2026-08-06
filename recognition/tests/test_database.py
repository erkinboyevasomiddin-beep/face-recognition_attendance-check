import sqlite3

import numpy as np
import pytest

from recognition.app.database import (
    FaceEmbeddingsDatabase,
    ImageEnrollmentRecord,
    LegacyDatabaseError,
)
from recognition.scripts.migrate_gallery import migrate


def test_gallery_storage_multiple_templates_and_reenrollment(tmp_path):
    database = FaceEmbeddingsDatabase(tmp_path / "gallery.db")
    image = ImageEnrollmentRecord("synthetic/a.jpg", "used", "test", 1, True)
    database.upsert_person(
        person_id="DEMO-STU-0001",
        display_name="Same Demo Name",
        class_name="5-01",
        embeddings=[np.array([3.0, 4.0]), np.array([0.0, 2.0])],
        usable_images=2,
        image_records=[image],
    )
    database.upsert_person(
        person_id="DEMO-STU-0002",
        display_name="Same Demo Name",
        class_name="5-01",
        embeddings=[np.array([1.0, 0.0])],
        usable_images=1,
        image_records=[image],
    )
    identities = database.load_identities()
    assert database.count_identities() == 2
    assert len(identities) == 3
    assert {item.person_id for item in identities} == {"DEMO-STU-0001", "DEMO-STU-0002"}
    assert all(np.linalg.norm(item.embedding) == pytest.approx(1.0) for item in identities)

    database.upsert_person(
        person_id="DEMO-STU-0001",
        display_name="Updated Demo Name",
        embeddings=[np.array([1.0, 1.0])],
        usable_images=1,
        image_records=[],
    )
    updated = [item for item in database.load_identities() if item.person_id == "DEMO-STU-0001"]
    assert len(updated) == 1 and updated[0].display_name == "Updated Demo Name"

    database.upsert_person(
        person_id="demo-stu-0001",
        display_name="Case-insensitive re-enrollment",
        embeddings=[np.array([1.0, 0.0])],
        usable_images=1,
        image_records=[],
    )
    assert database.count_identities() == 2


def test_gallery_rejects_invalid_enrollment(tmp_path):
    database = FaceEmbeddingsDatabase(tmp_path / "gallery.db")
    try:
        database.upsert_person("", "Demo", [np.ones(2)], 1, [])
    except ValueError as exc:
        assert "person_id" in str(exc)
    else:
        raise AssertionError("empty stable ID accepted")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"display_name": " ", "embeddings": [np.ones(2)]}, "display_name"),
        ({"display_name": "Demo", "embeddings": []}, "representative"),
        ({"display_name": "Demo", "embeddings": [np.ones(2), np.ones(3)]}, "same dimension"),
        ({"display_name": "Demo", "embeddings": [np.array([np.nan, 1.0])]}, "finite"),
    ],
)
def test_gallery_rejects_each_invalid_template_shape(tmp_path, kwargs, message):
    database = FaceEmbeddingsDatabase(tmp_path / "gallery.db")
    with pytest.raises(ValueError, match=message):
        database.upsert_person(
            person_id="DEMO-001",
            usable_images=1,
            image_records=[],
            **kwargs,
        )


def test_gallery_detects_legacy_and_corrupt_storage(tmp_path):
    legacy_path = tmp_path / "legacy.db"
    with sqlite3.connect(legacy_path) as connection:
        connection.execute("CREATE TABLE persons (id INTEGER PRIMARY KEY, name TEXT)")
    with pytest.raises(LegacyDatabaseError, match="version-1"):
        FaceEmbeddingsDatabase(legacy_path)

    database = FaceEmbeddingsDatabase(tmp_path / "corrupt.db")
    database.upsert_person(
        "DEMO-001",
        "Demo Person",
        [np.ones(2)],
        1,
        [],
    )
    with sqlite3.connect(database.db_path) as connection:
        connection.execute("UPDATE face_embeddings SET embedding_dim = 3")
    with pytest.raises(ValueError, match="size mismatch"):
        database.load_identities()


def test_gallery_rejects_unsupported_schema_version(tmp_path):
    path = tmp_path / "future.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE schema_info (version INTEGER NOT NULL)")
        connection.execute("INSERT INTO schema_info VALUES (999)")
    with pytest.raises(RuntimeError, match="Unsupported gallery schema"):
        FaceEmbeddingsDatabase(path)


def test_gallery_migration_preserves_source_and_applies_mapping(tmp_path):
    source = tmp_path / "legacy-gallery.db"
    destination = tmp_path / "gallery-v2.db"
    mapping = tmp_path / "mapping.csv"
    vector = np.array([3.0, 4.0], dtype=np.float32)
    with sqlite3.connect(source) as connection:
        connection.executescript(
            """
            CREATE TABLE persons (id INTEGER, name TEXT, class_name TEXT, embedding BLOB,
              embedding_dim INTEGER, usable_images INTEGER);
            CREATE TABLE source_images (id INTEGER, person_id INTEGER, image_path TEXT,
              status TEXT, note TEXT, face_count INTEGER, used_for_embedding INTEGER);
            """
        )
        connection.execute(
            "INSERT INTO persons VALUES (1, 'Legacy Demo', '5-01', ?, 2, 1)",
            (vector.tobytes(),),
        )
        connection.execute(
            "INSERT INTO source_images VALUES (1, 1, 'private/source.jpg', 'used', 'legacy', 1, 1)"
        )
    mapping.write_text(
        "legacy_name,person_id,display_name\nLegacy Demo,DEMO-001,Synthetic Demo\n",
        encoding="utf-8",
    )
    source_size = source.stat().st_size
    migrate(source, destination, mapping)
    assert source.stat().st_size == source_size
    identities = FaceEmbeddingsDatabase(destination).load_identities()
    assert identities[0].person_id == "DEMO-001"
    assert identities[0].display_name == "Synthetic Demo"
    assert np.linalg.norm(identities[0].embedding) == pytest.approx(1.0)

    with pytest.raises(FileExistsError):
        migrate(source, destination, mapping)


def test_gallery_mapping_requires_columns(tmp_path):
    source = tmp_path / "legacy.db"
    destination = tmp_path / "destination.db"
    mapping = tmp_path / "mapping.csv"
    source.write_bytes(b"placeholder")
    mapping.write_text("wrong,columns\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires"):
        migrate(source, destination, mapping)
