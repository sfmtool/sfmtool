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


def photograph_cache_lines(
    cache, paths: Iterable[Path], *workspace_dirs: str | Path
) -> list[str]:
    """The `Photographs:` line of ``cache``'s whole decodes, mipmaps made and
    budget, and, when it decoded the in-core photographs among ``paths`` (the
    photographs the command read) more times than there are of them, a line
    recommending a ``mipmap_options`` object in the ``.sfm-workspace.json`` of
    each of ``workspace_dirs``, the workspaces the photographs are in.

    A photograph decoded whole more than once had its tiles evicted while the
    command still read it: the photographs in core are more than the budget
    holds, and reading them out of core would decode none whole.
    """
    stats = cache.stats()
    lines = [
        f"Photographs: {stats['decodes']} whole decodes, "
        f"{stats['mipmaps_made']} mipmaps made, cache budget "
        f"{stats['budget_bytes'] / 2**30:.1f} GiB"
    ]
    in_core = sum(not cache.layout(path).out_of_core for path in set(paths))
    if stats["decodes"] > in_core:
        markers = dict.fromkeys(
            str(Path(workspace_dir) / ".sfm-workspace.json")
            for workspace_dir in workspace_dirs
        )
        lines.append(
            f"The cache decoded {in_core} photographs whole {stats['decodes']} "
            "times, since their tiles are more than its budget. To read them "
            'from mipmaps instead, a tile at a time, add a "mipmap_options" '
            f"object to {' and '.join(markers)}, such as "
            '"mipmap_options": {"out_of_core_long_side": 4096, '
            '"jpeg_quality": 95, "full_resolution_jpeg_quality": 95}.'
        )
    return lines
