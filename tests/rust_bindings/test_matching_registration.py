# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Registration coverage for `_sfmtool.matching`.

The matching submodule is wired up via per-file `pub fn register` blocks
that hand-list every `#[pyfunction]`. A typo or dropped line compiles
clean but ships an unreachable symbol; only an end-to-end test path that
happens to hit the dropped function would notice. This module asserts
every expected binding is present so a drop fails fast at collection time.
"""

import sfmtool._sfmtool.matching as matching

# The 12 symbols carried by the matching submodule. Mirrors the four per-file
# `pub fn register` bodies in `crates/sfmtool-py/src/matching/`.
_EXPECTED = (
    # cluster.rs
    "background_floor_clusters",
    "clusters_to_pair_matches",
    # descriptor.rs
    "descriptor_distance",
    "find_best_descriptor_match",
    # image.rs
    "match_image_pair_py",
    "match_image_pairs_batch_py",
    # sweep.rs
    "match_one_way_sweep_py",
    "match_one_way_sweep_geometric_py",
    "mutual_best_match_sweep_py",
    "polar_mutual_best_match_py",
    "mutual_best_match_sweep_geometric_py",
    "polar_mutual_best_match_geometric_py",
)


def test_all_matching_bindings_registered():
    """Every expected symbol is callable on `_sfmtool.matching`."""
    missing = [name for name in _EXPECTED if not hasattr(matching, name)]
    assert not missing, f"missing matching bindings: {missing}"
    for name in _EXPECTED:
        assert callable(getattr(matching, name)), f"{name} is not callable"


def test_matching_submodule_public_name():
    """The submodule reports its public `__name__` so binding objects'
    `__module__` reads `sfmtool.matching` in tracebacks, IPython, and Sphinx."""
    assert matching.__name__ == "sfmtool.matching"
    for name in _EXPECTED:
        assert getattr(matching, name).__module__ == "sfmtool.matching"
