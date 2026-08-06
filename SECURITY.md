# Security policy

## Supported versions

Security fixes are applied to the current default branch. This prototype has not completed an
independent security assessment and should not be exposed to untrusted networks as-is.

## Reporting a vulnerability

Do not open a public issue containing student/staff data, face images, embeddings, credentials,
private URLs, database copies, or exploit details. Before publication, the repository owner
must enable GitHub private vulnerability reporting and add the monitored security contact here.

Until that contact exists, keep the report private and include only synthetic reproduction
data, affected version/commit, impact, prerequisites, and a minimal proof of concept. Do not
access other people’s records, disrupt school systems, or retain personal data while testing.

## Operator responsibilities

- Keep the service private until TLS, unique secrets, exact allowed hosts, secure cookies,
  backups, retention, access control, and incident response are configured.
- Never commit `.env`, databases, embeddings, images, snapshots, models, logs, or real rosters.
- Rotate any key or password that may have been exposed and inspect Git history, forks,
  releases, caches, and CI artifacts; deleting the working-tree file is not enough.
- Review upstream Python/runtime/model advisories and the separate model license.
- Treat a recognition match as fallible evidence and preserve manual review/correction.

## Explicit limitations

No liveness detection, encryption-at-rest layer, distributed rate limiter, per-camera key,
immutable audit log, or formal penetration test is included. See
[`docs/security-and-privacy.md`](docs/security-and-privacy.md).
