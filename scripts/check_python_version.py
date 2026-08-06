from __future__ import annotations

import platform
import sys


def main() -> int:
    if sys.version_info[:2] == (3, 11):
        return 0

    detected = platform.python_version()
    print(
        "This project currently requires Python 3.11.x.\n"
        f"Detected Python {detected}.\n"
        "Create a Python 3.11 virtual environment and run the setup again.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
