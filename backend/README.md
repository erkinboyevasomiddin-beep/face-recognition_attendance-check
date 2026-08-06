# Backend

The backend is a FastAPI/Jinja application for roster, attendance, recognition-event, and
role-based dashboard workflows. It uses `person_id` values, UTC timestamps, Asia/Tashkent
display, Argon2id passwords, signed sessions, CSRF tokens, trusted hosts, explicit event
semantics, and SQLite foreign-key/WAL settings.

Python 3.11 is currently required for the project as a whole. Python 3.12 and newer are not
supported by the verified face-recognition dependency stack.

From the repository root:

```powershell
python -m backend.scripts.setup_demo
python -m uvicorn backend.main:app --reload
```

The demo command prints generated local director and teacher credentials once; it does not
publish or reuse a default password. Readiness is available at `/healthz`. See the root
[`README.md`](../README.md), [`docs/api.md`](../docs/api.md), and
[`docs/deployment.md`](../docs/deployment.md).
