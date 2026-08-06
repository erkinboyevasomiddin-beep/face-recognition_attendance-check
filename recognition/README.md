# Recognition client

The recognition client runs locally with InsightFace, ONNX Runtime, OpenCV, and a private
SQLite gallery. Enrollment metadata separates immutable `person_id` from `display_name`.
Webcam recognition uses per-face tracking and consecutive-hit confirmation, then sends an
explicit `ENTRY` or `EXIT` recognition event through a bounded background queue.

Use Python 3.11. Python 3.12 and newer are not supported by the verified InsightFace/NumPy
stack; see [`docs/python-compatibility.md`](../docs/python-compatibility.md).

```powershell
python -m pip install -c constraints-python311.txt -e ".[recognition]"
python -m recognition.main --mode enroll
python -m recognition.main --mode self-check
python -m recognition.main --mode webcam --camera-index 0
```

Models, face images, embeddings, snapshots, and evaluation data are local and ignored. There
is no liveness detection, and the provided InsightFace pretrained models have separate usage
restrictions. Read the root [`README.md`](../README.md),
[`docs/security-and-privacy.md`](../docs/security-and-privacy.md), and
[`docs/recognition-evaluation.md`](../docs/recognition-evaluation.md) before collecting data.
