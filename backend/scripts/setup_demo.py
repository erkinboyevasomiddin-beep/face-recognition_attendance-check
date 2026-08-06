from __future__ import annotations

import argparse
import os
import secrets
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"
DEMO_USERS = (
    ("demo.director", "Synthetic Director", "director"),
    ("demo.teacher", "Synthetic Teacher", "teacher"),
)


def _environment_value(text: str, key: str) -> str | None:
    prefix = f"{key}="
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def _replace_or_append(text: str, key: str, value: str) -> str:
    prefix = f"{key}="
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            current = line[len(prefix) :].casefold()
            if any(marker in current for marker in ("replace-with", "change-me", "placeholder")):
                lines[index] = f"{prefix}{value}"
            return "\n".join(lines) + "\n"
    lines.append(f"{prefix}{value}")
    return "\n".join(lines) + "\n"


def ensure_development_environment() -> bool:
    """Create a local environment file and replace only documented placeholders."""
    source = ENV_PATH if ENV_PATH.exists() else ENV_EXAMPLE_PATH
    text = source.read_text(encoding="utf-8")
    configured_environment = (
        os.environ.get("ATTENDANCE_APP_ENV")
        or _environment_value(text, "ATTENDANCE_APP_ENV")
        or "development"
    )
    if configured_environment == "production":
        raise RuntimeError("Synthetic demo setup is disabled in production")

    updated = _replace_or_append(
        text,
        "ATTENDANCE_SESSION_SECRET",
        secrets.token_urlsafe(48),
    )
    updated = _replace_or_append(
        updated,
        "ATTENDANCE_API_INGEST_KEY",
        secrets.token_urlsafe(48),
    )
    ingest_key = _environment_value(updated, "ATTENDANCE_API_INGEST_KEY")
    if ingest_key:
        updated = _replace_or_append(updated, "RECOGNITION_API_KEY", ingest_key)
    if not ENV_PATH.exists() or updated != text:
        ENV_PATH.write_text(updated, encoding="utf-8", newline="\n")
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a private local synthetic demo with generated credentials."
    )
    parser.parse_args()
    environment_changed = ensure_development_environment()

    # Import after .env exists so the cached typed settings use the generated values.
    from sqlalchemy import select

    from backend.app.auth import hash_password
    from backend.app.database import SessionLocal, init_db
    from backend.app.models import Student, User
    from backend.app.roster_import import import_roster_file
    from backend.app.schemas import UserRole
    from backend.seed_data import seed_synthetic_data_if_requested

    init_db()
    created_credentials: list[tuple[str, str, str]] = []
    with SessionLocal() as db:
        non_demo_student = db.scalar(
            select(Student.id).where(~Student.person_id.startswith("DEMO-")).limit(1)
        )
        non_demo_user = db.scalar(
            select(User.id).where(~User.username.startswith("demo.")).limit(1)
        )
        if non_demo_student is not None or non_demo_user is not None:
            raise RuntimeError("Refusing to mix synthetic setup with an existing non-demo database")

        seed_synthetic_data_if_requested(db)
        import_roster_file(db, PROJECT_ROOT / "sample_data" / "synthetic_roster.csv")

        for username, display_name, role_value in DEMO_USERS:
            if db.scalar(select(User).where(User.username == username)) is not None:
                continue
            password = secrets.token_urlsafe(18)
            db.add(
                User(
                    username=username,
                    display_name=display_name,
                    role=UserRole(role_value),
                    password_hash=hash_password(password),
                )
            )
            created_credentials.append((username, password, role_value))
        db.commit()

    print("Synthetic demo database is ready.")
    if environment_changed:
        print("Generated local secrets in .env (the file is ignored by Git).")
    if created_credentials:
        print("Local synthetic credentials (shown once):")
        for username, password, role in created_credentials:
            print(f"  {role}: {username} / {password}")
        print("Store these only for the local demo; they are not recoverable from this script.")
    else:
        print("Synthetic accounts already exist; their passwords were not changed or displayed.")
    print("Start the backend with: python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
