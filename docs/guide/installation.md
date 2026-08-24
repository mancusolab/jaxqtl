# Installation

jaxQTL requires Python 3.11 or newer. Install the current release from PyPI:

```bash
pip install jaxqtl
```

With [uv](https://docs.astral.sh/uv/), add jaxQTL to an existing project:

```bash
uv add jaxqtl
```

Confirm that the command-line interface is available:

```bash
jaxqtl --help
```

The CLI uses the CPU backend by default. Select another installed JAX backend with
`--platform gpu` or `--platform tpu` on a mapping command.

For a source checkout and contributor setup, see [Development setup](../contributing.md).
