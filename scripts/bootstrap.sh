#!/usr/bin/env bash
set -euo pipefail

groups=()
force=0
extract=0
build_tables=0
skip_download=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group)
      groups+=("$2")
      shift 2
      ;;
    --force)
      force=1
      shift
      ;;
    --extract-archives)
      extract=1
      shift
      ;;
    --build-tables)
      build_tables=1
      shift
      ;;
    --skip-download)
      skip_download=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ${#groups[@]} -eq 0 ]]; then
  groups=("mvp_core")
fi

download_args=("scripts/download_data.py")
for group in "${groups[@]}"; do
  download_args+=("--group" "$group")
done
if [[ $force -eq 1 ]]; then
  download_args+=("--force")
fi

if [[ $skip_download -eq 0 ]]; then
  python "${download_args[@]}"
fi

if [[ $extract -eq 1 ]]; then
  extract_args=("scripts/extract_archives.py")
  for group in "${groups[@]}"; do
    extract_args+=("--group" "$group")
  done
  if [[ $force -eq 1 ]]; then
    extract_args+=("--force")
  fi
  python "${extract_args[@]}"
fi

if [[ $build_tables -eq 1 ]]; then
  python scripts/parse_refseq_viral.py
  python scripts/normalize_taxonomy_hosts.py
  python scripts/build_training_index.py
fi

python scripts/build_inventory.py
