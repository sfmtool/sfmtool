# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The lines a command that reads photographs through a photograph cache
prints once its reads are done: `sfm cluster-patches`, `sfm embed-patches`,
the `sfm xform` steps that read photographs, and the strip montages of
`sfm compare --strips` and `sfm inspect --strips`.

See `specs/cli/image-feature/cluster-patches-command.md` § "Process" and the
out-of-core draft's open question 21.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def photograph_cache_lines(cache, paths: Iterable[Path]) -> list[str]:
    """The `Photographs:` line of ``cache``'s whole decodes, mipmaps made and
    budget, and, when it decoded the in-core photographs among ``paths`` (the
    photographs the command read) more times than there are of them, a line
    saying what puts them out of core in each workspace they are in.

    A photograph decoded whole more than once had its tiles evicted while the
    command still read it: the photographs in core are more than the budget
    holds, and reading them out of core would decode none whole. For a
    workspace without ``mipmap_options`` the line gives the
    ``sfm ws mipmaps --enable`` command that adds the object, with
    ``--out-of-core-long-side`` when the default threshold would leave one of
    the photographs in core; for a workspace whose object leaves them in core,
    the threshold to set in its file.
    """
    stats = cache.stats()
    lines = [
        f"Photographs: {stats['decodes']} whole decodes, "
        f"{stats['mipmaps_made']} mipmaps made, cache budget "
        f"{stats['budget_bytes'] / 2**30:.1f} GiB"
    ]
    # Each in-core photograph's longer side.
    in_core = {}
    for path in dict.fromkeys(Path(p) for p in paths):
        layout = cache.layout(path)
        if not layout.out_of_core:
            in_core[path] = max(layout.width, layout.height)
    if stats["decodes"] > len(in_core):
        lines.append(
            f"The cache decoded {len(in_core)} photographs whole "
            f"{stats['decodes']} times, since their tiles are more than its "
            "budget." + _out_of_core_advice(in_core)
        )
    return lines


def _out_of_core_advice(in_core: dict[Path, int]) -> str:
    """The sentence saying how to put the photographs of ``in_core``, each
    with its longer side, out of core in each workspace they are in. A
    photograph is out of core when a side is over the workspace's threshold,
    so the threshold that puts a workspace's photographs out of core is one
    less than the shortest of their longer sides."""
    from sfmtool._cli_utils import shown_path
    from sfmtool.workspace import (
        find_workspace_for_path,
        new_workspace_mipmap_options,
        read_mipmap_options,
    )

    # By workspace root; the roots are named in sorted order, since a caller
    # may pass its paths as a set.
    shortest: dict[Path, int] = {}
    for path, side in in_core.items():
        root = find_workspace_for_path(path)
        if root is not None:
            shortest[root] = min(side, shortest.get(root, side))
    default = new_workspace_mipmap_options()["out_of_core_long_side"]
    steps = []
    for root, side in sorted(shortest.items()):
        threshold = side - 1
        options = read_mipmap_options(root)
        if options is None:
            option = (
                "" if threshold >= default else f"--out-of-core-long-side {threshold} "
            )
            steps.append(f"run 'sfm ws mipmaps --enable {option}{shown_path(root)}'")
        else:
            steps.append(
                f'set "out_of_core_long_side" in '
                f"{shown_path(root / '.sfm-workspace.json')} to {threshold} or "
                f"less (it is {options['out_of_core_long_side']})"
            )
    if not steps:
        return ""
    return (
        f" To read them from mipmaps instead, a tile at a time, {' and '.join(steps)}."
    )
