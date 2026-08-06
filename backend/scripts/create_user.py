from __future__ import annotations

import argparse
import getpass
import secrets

from sqlalchemy import select

from backend.app.auth import hash_password
from backend.app.database import SessionLocal, init_db
from backend.app.models import User
from backend.app.schemas import UserRole


def create_user(username: str, display_name: str, role: UserRole, password: str) -> None:
    init_db()
    with SessionLocal() as db:
        if db.scalar(select(User).where(User.username == username)):
            raise ValueError(f"Username already exists: {username}")
        db.add(
            User(
                username=username,
                display_name=display_name,
                role=role,
                password_hash=hash_password(password),
            )
        )
        db.commit()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an Argon2id-hashed application user.")
    parser.add_argument("username")
    parser.add_argument("--display-name", required=True)
    parser.add_argument(
        "--role",
        choices=tuple(role.value for role in UserRole),
        required=True,
    )
    parser.add_argument(
        "--generate-password",
        action="store_true",
        help="Generate and print a one-time random password instead of prompting.",
    )
    args = parser.parse_args()
    password = secrets.token_urlsafe(18) if args.generate_password else getpass.getpass()
    role = UserRole(args.role)
    create_user(args.username.strip(), args.display_name.strip(), role, password)
    print(f"Created {role.value} user {args.username!r}.")
    if args.generate_password:
        print(f"One-time password: {password}")
        print("Store it securely and change it before any non-demo use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
