# Contributing to LightlyTrain

## Development

```
git clone https://github.com/lightly-ai/lightly-train.git
uv venv
make install-dev
source .venv/bin/activate
```

Make sure the environment is activated before running the following commands.

> [!WARNING]\
> Prepending commands with `uv run` might not work properly. Activate the environment directly instead.

```
make format
make static-checks
make test
```

### Documentation

Documentation is in the [docs](./docs) folder. To build the documentation, install
dev dependencies with `make install-dev`, then move to the `docs` folder and run:

```
make docs
```

This builds the documentation in the `docs/build/<version>` folder.

To build the documentation for the stable version, checkout the branch with the
stable version and run:

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
