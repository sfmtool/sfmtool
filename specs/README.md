# Design Specifications

The authoritative design documents for sfmtool. Read the relevant spec before
making a non-trivial change, and update it when behaviour diverges.

Each directory mirrors the code it describes, so the path to a spec is
predictable from the path to its implementation.

New specs start from [TEMPLATE.md](TEMPLATE.md), whose default section order is
purpose, the public Rust interface and why it is shaped that way, the theory,
then implementation notes. It is a starting point rather than a form — depart
from it where the subject is served better. What holds either way: the opening
paragraph reads for someone who has read no other spec, a caller can find out
what to call without wading through a derivation, and a spec describes what the
code *is* — write a change proposal in `drafts/` and convert it before filing.

Location encodes lifecycle, so a standing spec carries no `**Status:**` line: a
draft in `drafts/` opens with `**Status:** Draft`, and everything else opens with
its purpose paragraph. A part that is specified but unbuilt is a present-tense
sentence in the standing spec plus an amendment draft it links, never an inline
marker. The pointer at the implementing code lives in the spec's interface
section, with repo paths written as relative Markdown links.

| Directory | Describes | Organized by |
|-----------|-----------|--------------|
| [cli/](cli/README.md) | Every `sfm` subcommand — flags, behaviour, output | the `--help` category the command is registered under in `cli.py` |
| [core/](core/README.md) | The algorithms in `sfmtool-core`, and the Python pipelines that drive them | one subdirectory per `crates/sfmtool-core/src/` module |
| [formats/](formats/README.md) | The on-disk file formats | one file per format crate |
| [gui/](gui/README.md) | The SfM Explorer viewer (`sfm-explorer`) | flat |
| [workspace/](workspace/README.md) | Workspace layout and its config files | flat |
| `drafts/` | Scratch space for specs not yet ready to file | — |
