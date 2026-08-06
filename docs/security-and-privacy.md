# Security, privacy, and responsible use

This is an attendance prototype, not secure biometric authentication. Face recognition can
produce false accepts and false rejects; attendance decisions must remain reviewable.

## Required governance before deployment

- Obtain specific, informed consent from students/guardians and staff where required.
- Document the attendance purpose and prohibit reuse for discipline, profiling, covert
  surveillance, or unrelated monitoring.
- Determine the applicable biometric, child-data, education-record, employment, and privacy
  rules with qualified local reviewers and school leadership. This is not legal advice.
- Offer a practical non-biometric attendance option and a way to withdraw consent without
  retaliation where policy/law requires it.
- Tell people what is collected, why, retention periods, access roles, error risks, and how to
  request access, correction, appeal, or deletion.

## Data minimization and access

Keep raw enrollment images, embeddings, optional snapshots, roster data, attendance, and
audit records on access-controlled systems. The recognition API needs a stable ID and event
metadata; it does not need to receive an embedding or face crop. Give gallery access only to
operators who manage enrollment. Give attendance access only to authorized school roles.
Backups inherit the same sensitivity and deletion obligations.

Snapshots are disabled by default. If enabled, define a short retention period, automatic
deletion process, incident access log, and clear justification. Do not use filenames as the
only access control. Delete a person by removing their ignored enrollment images, their local
gallery row/templates/source-image metadata, relevant snapshots, and backend records according
to the approved retention policy; verify backups and replicas separately.

## Liveness and presentation attacks

No liveness detector is implemented. The camera may accept printed photos, faces shown on a
phone, or replayed video. Temporal confirmation only checks repeated model output; it is not
liveness detection. Do not describe it as anti-spoofing and do not use the system to unlock
doors, accounts, payments, examinations, or other high-security actions.

## Bias and error handling

Model behavior may differ with skin tone, age, gender presentation, disability, camera angle,
lighting, image quality, pose, occlusion, and other factors. Upstream benchmark numbers do not
establish performance for this school. Measure consented local cohorts, avoid publishing tiny
identifiable subgroups, and involve affected people in reviewing the process. A human must be
able to inspect events and correct attendance without treating the model score as proof.

## Technical controls present

- Argon2id password hashing and no shared default production passwords.
- Signed HttpOnly sessions, SameSite controls, production Secure cookies, CSRF protection,
  trusted hosts, security headers, server-side roles, and process-local login throttling.
- API key comparison, disabled-by-default ingestion, event UUIDs, age validation, cooldown,
  input limits, explicit event types, and audited manual corrections.
- Ignored databases, images, embeddings, snapshots, models, environment files, logs, and
  evaluation data.

## Residual risks

The login limiter is in-memory and per process. API keys are shared secrets rather than
per-device credentials. SQLite and local files rely on host access controls and are not
encrypted by this project. There is no key rotation UI, retention scheduler, immutable audit
store, liveness system, security monitoring service, or formal threat-model review. HTTPS must
be supplied by a reverse proxy in production.

## Incident and deletion workflow

Stop ingestion, preserve only necessary logs, rotate affected secrets, identify exposed data,
follow the school’s incident process and applicable notification rules, and document actions.
For deletion, use a reviewed request tied to `person_id`, record authorization, remove each
approved data category, and confirm completion to the requester without exposing other people.
