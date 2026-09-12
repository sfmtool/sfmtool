# Grouping nonconsecutive shared descriptor reads

**Status:** Draft. Amends
[core/features/lazy-kdforest-query.md](../core/features/lazy-kdforest-query.md),
"Current access path and performance diagnosis".

The shared query path borrows vectors from a pinned descriptor block and reuses
that pin across consecutive requests in the same block. It releases the tree pin
before admitting a descriptor block. It does not reorder leaf candidates.

A remaining experiment is to group all checked candidates in a leaf by block,
compute distances while holding one block at a time, then replay the distances in
the original leaf order. Replaying is required: the bounded result set retains the
first candidates encountered when distances tie. Sorting candidates and feeding
them directly to the result set changes answers.

The experiment needs bounded reusable scratch for the permutation and distances.
It must preserve indices, distances, check counts, and completion under a one-item
cache budget. Measure resident and cache-pressure workloads at equal worker counts;
include the sorting and replay costs. Reduced cache-hit counts alone are not a
reason to adopt it, and no performance improvement is assumed in advance.
