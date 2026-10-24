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

The pull request titles since the last release decide whether a release pull request
opens and which version it proposes. The largest bump wins.

| Type                                | Changelog section | Cuts a release | Bump below 1.0.0 |
| ----------------------------------- | ----------------- | -------------- | ---------------- |
| `feat`                              | Added             | yes            | patch            |
| `fix`                               | Fixed             | yes            | patch            |
| `perf`, `revert`                    | Changed           | yes            | patch            |
| `refactor`, `docs`, `test`, `build` | hidden            | no             | none             |
| `ci`, `chore`                       | hidden            | no             | none             |
| any type with `!`                   | Breaking Changes  | yes            | minor            |
| `Release-As: X.Y.Z` (see below)     | none              | yes            | `X.Y.Z`          |

- Minor versions are story releases. They only happen with a breaking change or with
  `Release-As`.
- Hidden types still ship with the next release, they are only not listed.
- User-facing dependency changes, for example a higher minimum version, use `fix(deps):`
  or `feat!:`, not `build:`.
- From `1.0.0` on, `!` bumps the major version and `feat` the minor version.

### Release-As

Forces the version of the next release.

- Add `Release-As: X.Y.Z` as the last line of the squash commit message in the merge
  dialog, on its own line and without a leading `*`.
- Applies only to the next release.
- Check: the release pull request title shows `X.Y.Z`.

## Patch Releases

No manual steps. Merge the release pull request. The changelog consists of the pull
request titles, so write titles for users.

## Story Releases

For minor versions and `1.0.0`:

1. Open a story pull request on `main`, for example
   `docs: add news for LightlyTrain 0.19.0`:
   - Add the entry to "News" in [README.md](./README.md) and
     [docs/source/index.md](./docs/source/index.md).
   - Merge it with `Release-As: 0.19.0`.
1. Wait until the release pull request shows `0.19.0`. When ready to release, stop
   merging to `main`.
1. Polish the release pull request:
   - Push a commit to its branch that rewrites the `0.19.0` section of
     [CHANGELOG.md](./CHANGELOG.md).
   - Paste the same text into the pull request description.
1. Merge the release pull request.

### Editing the Release Pull Request

- The description between the two `---` lines becomes the GitHub release notes and the
  Discord message.
- Keep the `## [X.Y.Z](...) (YYYY-MM-DD)` header and both `---` lines. Otherwise
  release-please creates no release.
- Every merge to `main` regenerates the branch and the description. Edits are lost.

## Failures

The tag and the GitHub release exist before the tests run. If a step in
`lightly-train-private` fails:

- **Tests fail:** nothing is published. Fix forward on `main` and merge the next release
  pull request.
- **Infrastructure failure:** rerun the `Release` workflow in `lightly-train-private`
  via `workflow_dispatch` with the tag. Steps that are already done are safe to repeat.

The full runbook is `RELEASE.md` in `lightly-train-private`.

## Configuration

- [release-please-config.json](./release-please-config.json): changelog sections and
  release pull request text.
- [.release-please-manifest.json](./.release-please-manifest.json): current version.
  Updated by release-please.
- Lightly Release Bot: GitHub App installed on `lightly-train` and
  `lightly-train-private` with read and write access to contents, issues and pull
  requests. `RELEASE_APP_ID` (variable) and `RELEASE_APP_PRIVATE_KEY` (secret) are
  stored in the `release` environment of this repository.
