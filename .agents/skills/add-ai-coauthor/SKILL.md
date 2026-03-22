---
name: add-ai-coauthor
description: Add or list repository-local Git co-author trailers for this sample repository. Use when the user wants to add Codex, Gemini, Claude, or human co-authors without editing hook files manually.
---

# Add AI co-author

Use this skill to add or inspect `ai.coauthor` entries in this repository.

## Workflow

1. If the user wants the default Codex trailer, run:
   - `scripts/manage.sh codex`
2. If the user provides an exact trailer entry, run:
   - `scripts/manage.sh add "Name <email@example.com>"`
3. If the user wants to remove an exact trailer entry, run:
   - `scripts/manage.sh remove "Name <email@example.com>"`
4. If the user wants to inspect the current list, run:
   - `scripts/manage.sh list`
5. After changing entries, show the current configured list.

## Notes

- This skill only edits repository-local Git config for `ai.coauthor`.
- It does not guess provider emails. For Gemini, Claude, or human co-authors, require the exact `Name <email>` string you want saved.
- The Git hook in `.githooks/prepare-commit-msg` remains the enforcement layer that appends every configured `ai.coauthor` trailer to future commits.
