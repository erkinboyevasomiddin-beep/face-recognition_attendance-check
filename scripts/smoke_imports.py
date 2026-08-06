from __future__ import annotations

from importlib import import_module
from importlib.metadata import version

MODULES = (
    "backend.main",
    "recognition.main",
    "cv2",
    "insightface",
    "numpy",
    "onnxruntime",
)

DISTRIBUTIONS = (
    "numpy",
    "insightface",
    "onnxruntime",
    "opencv-python",
    "fastapi",
    "sqlalchemy",
)


def main() -> int:
    for module_name in MODULES:
        import_module(module_name)
    for distribution_name in DISTRIBUTIONS:
        print(f"{distribution_name}=={version(distribution_name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
