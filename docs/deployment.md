# Deployment

## Local development

Follow the root README, keep `ATTENDANCE_APP_ENV=development`, bind Uvicorn to localhost, and
use Python 3.11 with a local ignored SQLite file. Python 3.12 and newer are not supported by
the verified face-recognition dependency stack. Do not expose the development server to an
untrusted network.

## Backend container

The desktop camera client is intentionally not containerized because Windows webcam/UI and
GPU passthrough would complicate the primary local workflow.

The Dockerfile uses `python:3.11-slim`, matching the supported project interpreter.

```powershell
python -m backend.scripts.setup_demo
docker compose config
docker compose build --no-cache
docker compose run --rm backend python -m backend.scripts.setup_demo
docker compose up -d backend
Invoke-RestMethod http://127.0.0.1:8000/healthz
```

The host setup command generates the ignored `.env` values consumed by Compose. The one-off
container command initializes the named volume and prints new synthetic credentials once.
Compose persists `/app/backend/data` in that volume and runs the backend as a non-root user.
For a production-like deployment:

- set `ATTENDANCE_APP_ENV=production`;
- set `ATTENDANCE_PUBLIC_BASE_URL` to the external HTTPS origin;
- use a unique 32+ character session secret and ingest key;
- enable `ATTENDANCE_SESSION_COOKIE_SECURE=true`;
- list exact public hostnames in `ATTENDANCE_ALLOWED_HOSTS`;
- terminate TLS at a maintained reverse proxy and forward only necessary headers;
- bind published ports/firewall rules narrowly;
- back up and test restore of the database under an approved retention policy;
- run migrations against a copy before changing an existing database;
- use one Uvicorn worker with SQLite. Move to a managed database and distributed login limiter
  before scaling to multiple workers/hosts.

The recognition client must use an HTTPS API URL in production. Do not embed credentials in
the URL; use the API-key setting. Rotate keys by updating backend and clients in a coordinated
window.

## Database migration

Version-1 backend and face-gallery migrations always write a new destination:

```powershell
python -m backend.scripts.migrate_v1_database --source C:\private\attendance.db --destination backend\data\attendance-v2.db
python -m recognition.scripts.migrate_gallery --source C:\private\face_embeddings.db --destination recognition\data\embeddings\face_embeddings-v2.db --mapping C:\private\gallery-mapping.csv
```

Legacy backend account hashes are copied only so records remain attributable. Recreate users
with `backend.scripts.create_user` before production and retire the legacy accounts/database.
The legacy gallery mapping should assign authoritative roster IDs.
