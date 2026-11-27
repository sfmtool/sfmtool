# The photograph cache's statistics in the Background Tasks panel (amendment)

**Status:** Draft

Amends [`../gui/background-tasks.md`](../gui/background-tasks.md) § "Placement",
which states that the panel shows the photograph cache's statistics in a
tooltip and points back here. The statistics are those of
`PhotographCache::stats`
([`../core/camera/photograph-cache.md`](../core/camera/photograph-cache.md)).

**What is to be built.** The Background Tasks panel shows the statistics of
the viewer's photograph cache (`AppState::photographs`) in a tooltip: the
tiles and file entries kept, their bytes and the budget, the hits, misses and
whole decodes, the mipmaps made, and the build budget with the bytes reserved
on it, the fields of `PhotographCacheStats`.

**The decision and its reason.** The maintainer decided to ship the cache's
default budgets (a quarter of physical memory, clamped to 1 to 16 GiB) and
Image Detail's 256 tile textures, and to tune them on real captures. Tuning
needs to see what the cache does while the viewer runs, so the decision was to
report the tiles decoded and reused in each kernel's phase note, which is
built (each step's `read regions` note and the `decode images` note's cache
occupancy), and the cache's `stats` in the Background Tasks panel's tooltip,
which is not. Where in the panel the tooltip sits and how it lays out the
figures is still to be chosen.
