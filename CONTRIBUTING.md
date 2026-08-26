# Contributing to LightlyTrain

## Development

### Setting up the Development Environment

```
git clone https://github.com/lightly-ai/lightly-train.git
make install-dev
```

#### Running MIGraphX Tests

Exporting and testing models with [MIGraphX](https://github.com/ROCm/AMDMIGraphX)
requires an AMD GPU with ROCm drivers, so this is not part of the regular development
setup above. Instead, use the provided VS Code dev container. Regular development should
not happen in this container — use `make install-dev` for that.

Prerequisites:

- VS Code with the
  [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
  extension
- Docker
- A host machine with an AMD GPU and ROCm-capable drivers

Open the repository in VS Code and run "Dev Containers: Reopen in Container" from the
command palette. This picks up `.devcontainer/migraphx/devcontainer.json`, which builds
an image based on `rocm/dev-ubuntu-24.04`, installs the `migraphx` apt package, passes
through the GPU devices, and runs `uv sync` with the `dev` and `pinned-rocm-torch`
groups plus the `onnx`, `onnxruntime`, and `onnxslim` extras.

Once inside the container, run the MIGraphX tests with:

```
uv run --frozen pytest tests/_export/test_export_migraphx.py tests/_task_models/ltdetr_object_detection/test_task_model.py -k migraphx -v
```

`test_export_migraphx` in `test_task_model.py` is skipped unless `migraphx`, `onnx`, and
`onnxruntime` are importable and `torch.cuda.is_available()` — i.e. it only actually
runs inside this container. To run just that test:

```
uv run --frozen pytest "tests/_task_models/ltdetr_object_detection/test_task_model.py::test_export_migraphx" -v
```

### Running Checks and Tests

Before committing code, make sure all tests and checks pass:

```
make format
make static-checks
```

and if you want to run all the tests:

```
make test
```

To run a specific test file, use:

```
uv run --frozen pytest path/to/test_file.py
```

### Pull Requests

Pull request titles must follow the
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.
This is checked by the `Lint PR Title` CI job:

```
<type>[optional scope][!]: <description>
```

Only the pull request title is checked. Pull requests are merged with GitHub's "Squash
and merge", so the title becomes the commit message on `main`. The individual commits on
your branch are squashed away and can be named freely.

Allowed types:

| Type       | Use for                                                      |
| ---------- | ------------------------------------------------------------ |
| `feat`     | A new user-facing feature.                                   |
| `fix`      | A bug fix.                                                   |
| `docs`     | Documentation only.                                          |
| `refactor` | A change that neither fixes a bug nor adds a feature.        |
| `perf`     | A change that improves performance.                          |
| `test`     | Adding or fixing tests.                                      |
| `build`    | Build system, packaging, or dependency changes.              |
| `ci`       | CI configuration and workflows.                              |
| `chore`    | Anything else that affects neither the source nor the tests. |
| `revert`   | Reverting a previous change.                                 |

The scope is optional and unrestricted. Everything after the colon is free-form, so
ticket references are fine:

```
feat(ltdetr): enforce 4 px min bbox guard in train transforms (TRN-2312)
fix: skip degenerate predicted boxes in val visualization instead of crashing
docs(settings): document nancapture debug utility
```

Mark breaking changes with a `!` before the colon, for example `feat(train)!: ...`.

The type determines how the version is bumped: `feat` bumps the minor version (0.18.0 →
0.19.0), `fix` and `perf` bump the patch version, and the remaining types don't trigger
a release on their own. While the version is below 1.0.0, breaking changes bump the
minor version instead of the major version.

Entries in [CHANGELOG.md](./CHANGELOG.md) are still written by hand. See
[RELEASING.md](./RELEASING.md) for the full release process.

### Documentation

Documentation is in the [docs](./docs) folder. To build the documentation, install dev
dependencies with `make install-dev`, then move to the `docs` folder and run:

```
make docs
```

This builds the documentation in the `docs/build/<version>` folder.

To build the documentation for the stable version, checkout the branch with the stable
version and run:

```
make docs-stable
```

This builds the documentaion in the `docs/build/stable` folder.

Docs can be served locally with:

```
make serve
```

#### Writing Documentation

The documentation source is in [docs/source](./docs/source). The documentation is
written in Markdown (MyST flavor). For more information regarding formatting, see:

- https://pradyunsg.me/furo/reference/
- https://myst-parser.readthedocs.io/en/latest/syntax/typography.html

### Contributor License Agreement (CLA)

To contribute to this repository, you must sign a Contributor License Agreement (CLA).
This is a one-time process done through GitHub when you open your first pull request.
You will be prompted automatically.

By signing the CLA, you agree that your contributions may be used under the terms of the
project license.
