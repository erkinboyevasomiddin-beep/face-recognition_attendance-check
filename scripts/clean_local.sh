#!/usr/bin/env bash
set -euo pipefail

script_dir="${BASH_SOURCE[0]%/*}"
[[ "$script_dir" == "${BASH_SOURCE[0]}" ]] && script_dir="."
project_root="$(cd "$script_dir/.." && pwd -P)"
apply=false
if [[ "${1:-}" == "--apply" ]]; then
  apply=true
elif [[ $# -gt 0 ]]; then
  printf 'Usage: %s [--apply]\n' "$0" >&2
  exit 2
fi

targets=(
  "$project_root/.pytest_cache"
  "$project_root/.mypy_cache"
  "$project_root/.ruff_cache"
  "$project_root/htmlcov"
  "$project_root/.coverage"
)

for target in "${targets[@]}"; do
  case "$target" in
    "$project_root"/*) ;;
    *) printf 'Refusing target outside project: %s\n' "$target" >&2; exit 1 ;;
  esac
  if [[ -e "$target" ]]; then
    if [[ "$apply" == true ]]; then
      rm -rf -- "$target"
      printf 'Removed %s\n' "$target"
    else
      printf 'Would remove %s\n' "$target"
    fi
  fi
done

if [[ "$apply" == false ]]; then
  printf '%s\n' 'Preview only. Re-run with --apply after reviewing the exact targets.'
fi
