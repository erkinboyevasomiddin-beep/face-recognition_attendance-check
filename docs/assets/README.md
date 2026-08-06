# Public visual assets

This directory intentionally contains no application screenshots yet. Add only images made
with the synthetic demo database and verify every visible label, browser autofill value,
notification, path, and terminal line before committing.

Expected files:

- `dashboard.png` — director dashboard at 1440×900 or similar, using only `DEMO-*` records.
- `recognition-demo.png` — recognition overlay using a consenting adult or a clearly synthetic
  illustration; never use a student photograph merely for documentation.
- `architecture.svg` — repository architecture diagram maintained alongside the Mermaid source.

## Screenshot procedure

1. Run `python -m backend.scripts.setup_demo` in a fresh local database.
2. Start the backend on localhost and sign in with the generated synthetic director account.
3. Hide browser bookmarks, password-manager prompts, local usernames, and unrelated windows.
4. Capture the dashboard and inspect the image at full resolution before saving it here.
5. Run the repository sensitive-data scan again before committing the image.

## Demo-video checklist

Record a short sequence showing synthetic setup, login, stable-ID roster rows, a mocked or
consented-adult recognition event, deterministic duplicate handling, and logout. Do not show
real faces, school records, credentials, `.env`, private paths, model directories, or ignored
runtime databases. State verbally or in a title card that liveness detection is absent.
