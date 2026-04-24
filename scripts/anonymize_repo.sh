#!/usr/bin/env bash
# Anonymize repository for double-blind submission (anon4open.org pipeline).
# Strips author names, internal paths, wandb entity, personal email from source tree.
#
# Usage: scripts/anonymize_repo.sh [output_dir]
# Default output: ./anon-release/

set -euo pipefail

OUTPUT_DIR="${1:-./anon-release}"
SOURCE_DIR="$(pwd)"

if [[ -d "$OUTPUT_DIR" ]]; then
    echo "[FAIL] Output directory already exists: $OUTPUT_DIR"
    exit 1
fi

echo "[INFO] Anonymizing $SOURCE_DIR -> $OUTPUT_DIR"

# Copy with selective exclusion
rsync -a \
    --exclude='.git/' \
    --exclude='data/' \
    --exclude='checkpoints/' \
    --exclude='log_files/' \
    --exclude='runs/' \
    --exclude='wandb/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='author_info.md' \
    --exclude='internal_notes/' \
    --exclude='.claude/' \
    --exclude='anon-release/' \
    "$SOURCE_DIR/" "$OUTPUT_DIR/"

# Strip author identifiers from Python + markdown files
# Extend this list as real names/handles are introduced.
ANON_PATTERNS=(
    's/[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/anon@example.com/g'
    's/thuanbui1309/anonuser/g'
    's|/home/[^/]+/projects/research/SCommander|/path/to/SCommander|g'
)

find "$OUTPUT_DIR" -type f \( -name '*.py' -o -name '*.md' -o -name '*.yaml' -o -name '*.toml' \) | while read -r f; do
    for pattern in "${ANON_PATTERNS[@]}"; do
        sed -i -E "$pattern" "$f"
    done
done

echo "[OK] Anonymized release staged at: $OUTPUT_DIR"
echo "[NEXT] Inspect, then zip and upload to https://anon4open.org/"
