# API

## Recognition ingestion

`POST /api/recognition-event`

Headers:

- `Content-Type: application/json`
- `X-API-Key: <configured ingest key>`
- `Idempotency-Key: <event UUID>` (recommended; when present it must match `event_id`)

Synthetic request:

```json
{
  "event_id": "9df044c2-931e-4c14-a17f-26a522bd13cf",
  "person_id": "DEMO-STU-0001",
  "display_name": "Demo Student 001",
  "class_name": "5-01",
  "similarity": 0.82,
  "timestamp": "2026-08-06T08:05:00+05:00",
  "event_type": "ENTRY",
  "source": "front-entry-camera"
}
```

The backend authenticates the client, validates shape/ranges/time zone and event age, applies
its own similarity threshold, looks up `person_id`, checks event UUID and cooldown, then
records the event. `display_name` is diagnostic/display metadata and never the primary key.

Outcomes:

- `applied`: attendance arrival or departure changed.
- `recorded`: valid observation recorded but no state change was needed.
- `duplicate`: the same event UUID was already processed.
- `cooldown`: a recent same-person/type/source event suppressed a change.
- `unmatched`: stable ID was not in the roster.
- `rejected`: similarity was below the backend threshold.

HTTP `401` means the API key failed, `400` means an age/idempotency validation failed, `422`
means the JSON schema failed, and `404` is returned when API ingestion is disabled.

## Interactive routes

`/login`, `/logout`, `/director/*`, and `/teacher/*` use an HttpOnly signed session cookie.
State-changing browser forms require a session-bound CSRF token. Director and teacher roles
are checked by route dependencies on the server.

`POST /director/attendance/correction` is director-only and records the actor, previous and
new status, reason, and timestamp in `manual_corrections`.

## Health

`GET /healthz` performs a lightweight database query and returns `{"status":"ok"}`. It is
unauthenticated so local process/container health checks can use it; it contains no version,
path, database, roster, or secret information.

See [`attendance-state-machine.md`](attendance-state-machine.md) for transition semantics.
