#!/usr/bin/env bash
set -euo pipefail

script_dir="${BASH_SOURCE[0]%/*}"
[[ "$script_dir" == "${BASH_SOURCE[0]}" ]] && script_dir="."
project_root="$(cd "$script_dir/.." && pwd -P)"
python_bin="${PYTHON_BIN:-python3}"
install_recognition=false
setup_synthetic=false

for option in "$@"; do
  case "$option" in
    --recognition) install_recognition=true ;;
    --synthetic) setup_synthetic=true ;;
    *) printf 'Unknown option: %s\n' "$option" >&2; exit 2 ;;
  esac
done

"$python_bin" -m venv "$project_root/.venv"
source "$project_root/.venv/bin/activate"
python -m pip install 'pip>=26.1.2'
python -m pip install -e "$project_root[dev]"
if [[ "$install_recognition" == true ]]; then
  python -m pip install -e "$project_root[recognition]"
fi

if [[ ! -f "$project_root/.env" ]]; then
  cp "$project_root/.env.example" "$project_root/.env"
  printf '%s\n' 'Created .env from the documented development template.'
fi

if [[ "$setup_synthetic" == true ]]; then
  cd "$project_root"
  python -m backend.scripts.setup_demo
fi
