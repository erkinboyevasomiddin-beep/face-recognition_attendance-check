#!/usr/bin/env bash
set -euo pipefail

script_dir="${BASH_SOURCE[0]%/*}"
[[ "$script_dir" == "${BASH_SOURCE[0]}" ]] && script_dir="."
project_root="$(cd "$script_dir/.." && pwd -P)"
source "$project_root/.venv/bin/activate"
cd "$project_root"
python scripts/check_python_version.py
python -m ruff check .
python -m ruff format --check .
python -m mypy backend recognition
python -m pytest
python -m pip check
python -m pip_audit --cache-dir "$project_root/.local/pip-audit-cache" --skip-editable
