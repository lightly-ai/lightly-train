---
name: pr-self-review-protocol
description: "Use when reviewing your own LightlyTrain changes before opening a PR."
---

# PR Self-Review Protocol

- Review the final diff as if you were the reviewer.
- Confirm the PR answers: what changed, why, how it was validated, and what remains risky.
- Check whether user-facing changes need docs, `CHANGELOG.md`, or examples.
- Check whether generated files or mirrored docs were updated at the source of truth.
- Verify tests cover the exact behavior changed; prefer targeted tests.
- Look for edge cases: empty inputs, missing labels, platform differences, API mismatches, licensing, and build/release implications.

## Self-review questions
- Would this be confusing out of context?
- Is there a simpler or more local way to express this?
- Did I update docs/changelog/examples where needed?
- Did I introduce new files, copied code, or third-party content that needs licensing review?
- What reviewer question is still unanswered?

## Rules
- Do not open a PR until obvious reviewer concerns are addressed.
- If a review comment is likely, fix it before asking for review.
- Keep the PR description aligned with the actual diff and validation.
