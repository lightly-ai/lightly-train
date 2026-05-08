---
name: development-workflow-protocol
description: "Use when planning, developing, validating, and preparing LightlyTrain changes for commit/PR."
---

# Development Workflow Protocol

- Start by identifying the touched codepaths, docs, tests, and CI jobs.
- Work in small increments.
- Always use the active `.venv`.
- Use `make format` while developing, then `make format-check` before commit.
- Use targeted `pytest` before broader suites.

## Typical workflow
1. Plan: inspect relevant source, docs, tests, and workflow files.
2. Implement: make a small change, format it, and run the most relevant test.
3. Validate: run the matching repo checks.
4. Polish: update docs, changelog, and examples when user-facing.
5. Commit: only after the relevant checks pass or the remaining risk is understood.

## Check map
- Python / source changes: `make format`, `make type-check`, targeted `pytest path/to/test_file.py`, then `make format-check`.
- Docs / markdown changes: `make format`, `make format-check`; if rendered output matters, run `cd docs && make docs`.
- User-facing changes: update docs and `CHANGELOG.md` when appropriate.
- Dependency / packaging changes: run the matching install target first (`make install-dev`, `make install-docs`, `make install-minimal`, `make install-minimal-extras`, `make install-pinned-3.8`, or `make install-pinned-3.13`), then the relevant validation.
- Build / release prep: run `make install-dist` and `make dist`.
- Broad changes: prefer `make static-checks` and `make test` when the scope justifies it.

## Rules
- Prefer the smallest check set that covers the change.
- If a change spans multiple workflows, run every affected check.
- Do not stage or commit until the relevant checks are done.
