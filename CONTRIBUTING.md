# Contributing

Thanks for your interest in improving this project.

Contributions are welcome, including:
- bug fixes
- calibration workflow improvements
- documentation updates
- performance and tracking quality improvements

## How to contribute
1. Fork the repository and create a feature branch.
2. Make focused changes with clear commit messages.
3. Run tests before opening a PR:
   ```bash
   python -m unittest discover -s tests
   ```
4. Open a Pull Request with:
   - a short problem statement
   - what changed
   - how you tested it

## Style and scope
- Keep changes minimal and targeted.
- Update docs when behavior or CLI options change.
- Prefer backward-compatible defaults.

## Reporting issues
When opening an issue, please include:
- input video details (resolution/FPS)
- config file (or relevant sections)
- command used to run
- observed vs expected behavior
