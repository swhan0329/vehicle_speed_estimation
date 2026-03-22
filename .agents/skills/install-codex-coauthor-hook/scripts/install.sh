#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
skill_dir=$(cd "$script_dir/.." && pwd)
repo_root=$(cd "$skill_dir/../../.." && pwd)
setup_script="$repo_root/scripts/setup-codex-attribution.sh"

if [[ ! -x "$setup_script" ]]; then
  echo "Expected setup script at $setup_script" >&2
  exit 1
fi

"$setup_script"

echo
echo "Verified repository-local Git settings:"
git -C "$repo_root" config --local --get core.hooksPath
git -C "$repo_root" config --local --get commit.template
git -C "$repo_root" config --local --get-all ai.coauthor
