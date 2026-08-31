# Design Specifications

The authoritative design documents for sfmtool. Read the relevant spec before
making a non-trivial change, and update it when behaviour diverges.

Each directory mirrors the code it describes, so the path to a spec is
predictable from the path to its implementation.

New specs start from [TEMPLATE.md](TEMPLATE.md), which fixes the section order —
purpose, the public Rust interface and why it is shaped that way, the theory,
then implementation notes. A spec describes what the code *is*; write a change
proposal in `drafts/` and convert it before filing.

| Directory | Describes | Organized by |
|-----------|-----------|--------------|
| [cli/](cli/README.md) | Every `sfm` subcommand — flags, behaviour, output | the `--help` category the command is registered under in `cli.py` |
| [core/](core/README.md) | The algorithms in `sfmtool-core`, and the Python pipelines that drive them | one subdirectory per `crates/sfmtool-core/src/` module |
| [formats/](formats/README.md) | The on-disk file formats | one file per format crate |
| [gui/](gui/README.md) | The SfM Explorer viewer (`sfm-explorer`) | flat |
| [workspace/](workspace/README.md) | Workspace layout and its config files | flat |
| `drafts/` | Scratch space for specs not yet ready to file | — |
