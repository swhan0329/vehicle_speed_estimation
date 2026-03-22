#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
skill_dir=$(cd "$script_dir/.." && pwd)
repo_root=$(cd "$skill_dir/../../.." && pwd)

read_local_config() {
  local key=$1
  git -C "$repo_root" config --local --get "$key" 2>/dev/null || true
}

last_commit_body() {
  git -C "$repo_root" show -s --format=%B HEAD 2>/dev/null || true
}

printf 'Repository: %s\n' "$repo_root"
printf 'Hook path: %s\n' "$(read_local_config core.hooksPath)"
printf 'Commit template: %s\n' "$(read_local_config commit.template)"

printf '\nConfigured ai.coauthor entries:\n'
if ! git -C "$repo_root" config --local --get-all ai.coauthor 2>/dev/null; then
  printf '(none)\n'
fi

printf '\nLatest commit subject:\n'
git -C "$repo_root" show -s --format=%s HEAD 2>/dev/null || printf '(no commits)\n'

printf '\nLatest commit trailers:\n'
trailers=$(last_commit_body | grep '^Co-authored-by:' || true)
if [[ -n "$trailers" ]]; then
  printf '%s\n' "$trailers"
else
  printf '(none)\n'
fi
