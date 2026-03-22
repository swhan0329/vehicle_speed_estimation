#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
skill_dir=$(cd "$script_dir/.." && pwd)
repo_root=$(cd "$skill_dir/../../.." && pwd)

usage() {
  cat <<'EOF'
Usage:
  scripts/manage.sh list
  scripts/manage.sh codex
  scripts/manage.sh add "Name <email@example.com>"
  scripts/manage.sh remove "Name <email@example.com>"
EOF
}

list_entries() {
  git -C "$repo_root" config --local --get-all ai.coauthor || true
}

ensure_entry() {
  local entry=$1

  if git -C "$repo_root" config --local --get-all ai.coauthor | grep -Fqx "$entry"; then
    echo "Already configured: $entry"
    return
  fi

  git -C "$repo_root" config --local --add ai.coauthor "$entry"
  echo "Added: $entry"
}

remove_entry() {
  local entry=$1

  if ! git -C "$repo_root" config --local --get-all ai.coauthor | grep -Fqx "$entry"; then
    echo "Not configured: $entry"
    return
  fi

  git -C "$repo_root" config --local --fixed-value --unset-all ai.coauthor "$entry"
  echo "Removed: $entry"
}

case "${1-}" in
  list)
    list_entries
    ;;
  codex)
    ensure_entry "codex <codex@openai.com>"
    echo
    list_entries
    ;;
  add)
    entry=${2-}
    if [[ -z "$entry" ]]; then
      usage >&2
      exit 1
    fi
    if ! printf '%s\n' "$entry" | grep -Eq '.+ <[^<>[:space:]]+@[^<>[:space:]]+>$'; then
      echo "Expected entry format: Name <email@example.com>" >&2
      exit 1
    fi
    ensure_entry "$entry"
    echo
    list_entries
    ;;
  remove)
    entry=${2-}
    if [[ -z "$entry" ]]; then
      usage >&2
      exit 1
    fi
    remove_entry "$entry"
    echo
    list_entries
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
