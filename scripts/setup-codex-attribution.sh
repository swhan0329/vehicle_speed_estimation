#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"

git config --local core.hooksPath .githooks
git config --local commit.template .gitmessage

git config --local --unset-all ai.coauthor >/dev/null 2>&1 || true
git config --local --add ai.coauthor "codex <codex@openai.com>"

echo "Configured repository-local Codex co-author attribution."
echo "Current co-authors:"
git config --get-all ai.coauthor
