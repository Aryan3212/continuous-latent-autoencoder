Use `CODEBASE.md` as the quick reference; avoid rescanning the repo unless necessary.
Don't run any commands for training they are long running and should be monitored by the user.
After major changes update `CODEBASE.md` and `CHANGELOG.md`
All Python commands use `uv run python` (uv v0.10.9 is installed). Never use bare `python` or `python3`.
WSL environment — manifest paths originally used Windows backslashes; run `scripts/fix_manifest_paths.py` after any manifest changes.
