from __future__ import annotations

import argparse

from recognition.app.face_engine import FaceEngine


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the optional InsightFace runtime and configured model files."
    )
    parser.parse_args()
    engine = FaceEngine()
    engine.ensure_loaded()
    print(f"InsightFace loaded with providers: {', '.join(engine.providers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
