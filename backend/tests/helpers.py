from __future__ import annotations

import re

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.auth import hash_password
from backend.app.models import SchoolClass, Student, User
from backend.app.schemas import UserRole


def add_user(
    db: Session,
    *,
    username: str = "director.test",
    password: str = "correct horse battery staple",
    role: UserRole = UserRole.DIRECTOR,
) -> User:
    user = User(
        username=username,
        display_name=f"Synthetic {role.value.title()}",
        role=role,
        password_hash=hash_password(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def add_student(
    db: Session,
    *,
    person_id: str = "DEMO-STU-0001",
    display_name: str = "Demo Student 001",
    class_name: str = "5-01",
) -> Student:
    school_class = db.query(SchoolClass).filter_by(name=class_name).one_or_none()
    if school_class is None:
        grade, section = class_name.split("-", maxsplit=1)
        school_class = SchoolClass(grade=int(grade), section_code=section, name=class_name)
        db.add(school_class)
        db.flush()
    student = Student(
        person_id=person_id,
        display_name=display_name,
        grade=school_class.grade,
        class_name=school_class.name,
        class_id=school_class.id,
    )
    db.add(student)
    db.commit()
    db.refresh(student)
    return student


def login(
    client: TestClient,
    username: str,
    password: str,
):
    page = client.get("/login")
    match = re.search(r'name="csrf_token" value="([^"]+)"', page.text)
    assert match
    return client.post(
        "/login",
        data={"username": username, "password": password, "csrf_token": match.group(1)},
    )
