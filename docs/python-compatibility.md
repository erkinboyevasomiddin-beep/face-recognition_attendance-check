# Python compatibility

The complete project is supported on Python 3.11 only. Setup scripts reject other interpreter
versions before installing packages or creating local data.

## Incompatibility evidence

The blocking combination is InsightFace 0.7.3 and NumPy on newer Python versions:

- `insightface.app.FaceAnalysis.draw_on` calls `astype(np.int)`. Running that path with
  InsightFace 0.7.3 and NumPy 2.4.4 on Python 3.13.3 raises `AttributeError` because NumPy
  removed `np.int` after the 1.23 series.
- A binary-only pip resolution check on Python 3.12.13 could not install NumPy 1.23.5; the
  available compatible wheels started at NumPy 1.26. A Python 3.13.3 check likewise could not
  install 1.23.5; its available wheels started at NumPy 2.1.
- Unconstrained current releases also move the NumPy floor: OpenCV 4.13 requires NumPy 2 on
  Python 3.9 and newer, while Albumentations 2.0.8 and scikit-image 0.26 require NumPy 1.24 or
  newer. Those versions cannot be combined with NumPy 1.23.5.

InsightFace, ONNX Runtime, OpenCV, and the InsightFace Cython extension imported successfully
in the inspected Python 3.13 environment. That import-only result does not remove the runtime
failure above. No project code using behavior removed in Python 3.12 or 3.13 was found.

## Verified dependency set

Direct core versions are declared in `pyproject.toml`:

- Python 3.11
- NumPy 1.23.5
- InsightFace 0.7.3
- ONNX Runtime 1.22.1
- OpenCV 4.11.0.86
- FastAPI 0.141.1
- SQLAlchemy 2.0.51

`constraints-python311.txt` limits the compiled and image-processing transitive packages that
would otherwise select a newer NumPy requirement. Install from the repository root with:

```text
python -m pip install -c constraints-python311.txt -e ".[dev,recognition]"
```

Model files are not part of the environment and remain excluded from the repository.
