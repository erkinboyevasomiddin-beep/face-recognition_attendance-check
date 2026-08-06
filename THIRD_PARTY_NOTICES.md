# Third-party notices

The Apache-2.0 license in this repository covers this project’s source code only. Dependencies
remain under their own licenses.

## InsightFace

InsightFace library code is published upstream under the MIT License. Upstream separately
states that pretrained models it provides, including the auto/manual `buffalo_l` package, are
for non-commercial research and now directs users to contact its licensing channel for open
face-recognition model licensing. Those model files are ignored and are not redistributed by
this repository. A school deployment must review the current upstream terms and obtain any
required rights; the project’s Apache-2.0 license does not grant them.

- Project/license notice: https://github.com/deepinsight/insightface#license
- Python package/model notice: https://github.com/deepinsight/insightface/tree/master/python-package

## Other dependencies

The direct declarations were reviewed against the installed package metadata for the release
environment and their upstream project metadata where wheel metadata was incomplete:

| Dependency group | Reported license family |
|---|---|
| FastAPI, SQLAlchemy, Pydantic Settings, Argon2 CFFI | MIT |
| Starlette, Uvicorn, HTTPX2, Jinja2, ItsDangerous | BSD-3-Clause |
| Requests, python-multipart, tzdata, OpenCV Python wheels | Apache-2.0 |
| NumPy | BSD-3-Clause plus licenses for bundled components shown in its distribution |
| ONNX Runtime | MIT |
| pytest, Ruff, mypy, build and associated development tools | permissive upstream licenses |

Versions are constrained in `pyproject.toml`, but every resolved transitive dependency retains
its own terms. Before distributing binaries or containers, generate a fresh dependency
inventory for that resolved environment and preserve required notices. Optional proprietary or
replacement models must be reviewed separately. This summary is an engineering review, not a
legal opinion or guarantee.
