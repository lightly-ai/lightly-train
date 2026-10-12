# Releasing LightlyTrain

Releases are automated with
[release-please](https://github.com/googleapis/release-please). Merging the release pull
request is the only manual step. See [CONTRIBUTING.md](./CONTRIBUTING.md#pull-requests)
for the pull request title convention that this process builds on.

## Flow

1. Every push to `main` runs [release.yml](./.github/workflows/release.yml).
   release-please keeps a release pull request `chore(release): LightlyTrain X.Y.Z` open
   and updates it on every merge:
   - Bumps `__version__` in `src/lightly_train/__init__.py`.
   - Adds the changelog entry to [CHANGELOG.md](./CHANGELOG.md), generated from the pull
     request titles.
1. Merging the release pull request creates the `vX.Y.Z` tag and the GitHub release, and
   starts the `Release` workflow in the private `lightly-train-private` repository with
   a `repository_dispatch` event.
1. `Release` in `lightly-train-private` runs in this order:
   1. Validates that the tag is a published `vX.Y.Z` release.
   1. Runs the full unit tests on the tag.
   1. Publishes to PyPI. Skipped if the version is already on PyPI.
   1. Publishes the documentation and the DockerHub image in parallel.
   1. Reports to the `#releases` Slack channel and to Discord.

The publishing workflows stay in `lightly-train-private` because they hold the PyPI, GCP
and DockerHub credentials. This repository only holds the credentials of the Lightly
Release Bot GitHub App, in the `release` environment.

## Version Bumps

- `feat` bumps the minor version.
- `fix` and `perf` bump the patch version.
- Breaking changes (`!`) bump the minor version while the version is below `1.0.0`.
- Other types are not listed in the changelog and don't cut a release on their own.

To force a specific version, add `Release-As: X.Y.Z` to the body of a commit on `main`.

## Failures

The tag and the GitHub release exist before the tests run. If a step in
`lightly-train-private` fails:

- **Tests fail:** nothing is published. Fix forward on `main` and merge the next release
  pull request.
- **Infrastructure failure:** rerun the `Release` workflow in `lightly-train-private`
  via `workflow_dispatch` with the tag. Steps that are already done are safe to repeat.

## Configuration

- [release-please-config.json](./release-please-config.json): changelog sections and
  release pull request text.
- [.release-please-manifest.json](./.release-please-manifest.json): current version.
  Updated by release-please.
- Lightly Release Bot: GitHub App installed on `lightly-train` and
  `lightly-train-private` with read and write access to contents, issues and pull
  requests. `RELEASE_APP_ID` (variable) and `RELEASE_APP_PRIVATE_KEY` (secret) are
  stored in the `release` environment of this repository.
