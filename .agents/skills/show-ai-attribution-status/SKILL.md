---
name: show-ai-attribution-status
description: Show this repository's Git attribution status, including hook installation, configured co-authors, and recent commit trailers. Use when the user wants to verify whether Codex co-author attribution is active and working.
---

# Show AI attribution status

Use this skill to inspect whether AI co-author attribution is correctly configured in this repository.

## Workflow

1. Run `scripts/status.sh`.
2. Report:
   - `core.hooksPath`
   - `commit.template`
   - all configured `ai.coauthor` values
   - the most recent commit subject
   - any `Co-authored-by:` trailers on the most recent commit
3. If the hook or config is missing, recommend `$install-codex-coauthor-hook`.
4. If the co-author list needs changes, recommend `$add-ai-coauthor`.

## Notes

- This skill is read-only.
- It is intended as a quick health check after setup, after edits, or before pushing commits.
