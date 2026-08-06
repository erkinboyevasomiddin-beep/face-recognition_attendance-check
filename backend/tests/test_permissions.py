from __future__ import annotations

from backend.app.schemas import UserRole
from backend.tests.helpers import add_user, login


def test_director_and_teacher_routes_are_server_enforced(client, db):
    add_user(db, username="director.test", role=UserRole.DIRECTOR)
    add_user(
        db,
        username="teacher.test",
        password="another correct horse password",
        role=UserRole.TEACHER,
    )

    assert login(client, "teacher.test", "another correct horse password").status_code == 303
    denied = client.get("/director")
    assert denied.status_code == 303
    assert denied.headers["location"] == "/teacher"
    assert client.get("/teacher").status_code == 200

    client.cookies.clear()
    assert login(client, "director.test", "correct horse battery staple").status_code == 303
    denied = client.get("/teacher")
    assert denied.status_code == 303
    assert denied.headers["location"] == "/director"
    events = client.get("/director/events")
    assert events.status_code == 200
    assert "Content-Security-Policy" in events.headers
