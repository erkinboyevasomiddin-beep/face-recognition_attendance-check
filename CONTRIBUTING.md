# Contributing

Contributions should preserve the split between the local recognition client and backend,
stable `person_id` identity model, explicit `ENTRY`/`EXIT` semantics, and synthetic-only tests.

## Development workflow

1. Open an issue describing the concrete defect or scoped change.
2. Create a branch and install `python -m pip install -e ".[dev]"`.
3. Add deterministic tests that need no camera, GPU, model download, private secret, or real
   school/biometric data.
4. Run `python -m ruff check .`, `python -m ruff format --check .`,
   `python -m mypy backend recognition`, and `python -m pytest`.
5. Explain schema/configuration/privacy impacts and include a copy-based migration when needed.

Do not submit real names, rosters, databases, face images, embeddings, recognition snapshots,
private IP addresses, machine paths, credentials, benchmark claims without data, or generated
model weights. Use the `DEMO-*` synthetic IDs and names in `sample_data/`.

Security reports belong in the private process described in [`SECURITY.md`](SECURITY.md), not
public issues. By contributing, you agree that your contribution is licensed under Apache-2.0.
