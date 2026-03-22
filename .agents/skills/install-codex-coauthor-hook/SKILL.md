---
name: install-codex-coauthor-hook
description: Install or repair the sample repository's Git prepare-commit-msg hook for automatic Codex co-author attribution. Use when the user wants Codex commit attribution enabled, restored, or reconfigured in this repository.
---

# Install Codex co-author hook

Use this skill when the task is to enable or repair automatic Codex co-author attribution in this repository.

## Workflow

1. Run `scripts/install.sh` from this skill directory.
2. Verify these repository-local Git settings:
   - `core.hooksPath=.githooks`
   - `commit.template=.gitmessage`
   - `ai.coauthor=codex <codex@openai.com>`
3. Report the configured co-authors and any next-step commands for adding more trailers.

## Notes

- This skill is repository-specific and delegates the actual setup to `scripts/setup-codex-attribution.sh`.
- The Git hook is the deterministic enforcement layer; this skill is a convenient installer/repair surface for Codex.
- If the user wants more co-authors, add them with `git config --local --add ai.coauthor "Name <email@example.com>"`.
