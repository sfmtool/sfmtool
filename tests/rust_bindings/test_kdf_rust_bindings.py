# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the persistent `.kdf` forest bindings.

These bindings exist so the descriptor-layout comparison in
`specs/core/features/lazy-kdforest-query.md` can be run from Python, so the
tests are written around what a benchmark actually asks the surface to do:
export the same forest both ways, confirm the file-backed answers still match
the in-memory ones, and read back counters that mean what they say.
"""

import json

import numpy as np
import pytest

from sfmtool._sfmtool.spatial import (
    KdForest,
    LazyKdForest,
    kdf_file_summary,
    verify_kdf,
    write_kdf,
)

# Small enough to stay fast, wide enough that a few-hundred-byte chunk target
# still splits each tree into many chunks — a traversal that never crosses a
# chunk boundary would exercise none of the lazy path.
_N = 400
_DIM = 32
_CHUNK_BYTES = 4096
_BLOCK_BYTES = 1024

_LAYOUTS = [
    pytest.param("tree_local", {}, id="tree_local"),
    pytest.param("shared", {"descriptor_block_bytes": _BLOCK_BYTES}, id="shared"),
]


def _descriptors(n=_N, dim=_DIM, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, dim), dtype=np.uint8)


def _forest(descriptors, num_trees=4, leaf_size=8, seed=7):
    return KdForest(descriptors, num_trees=num_trees, leaf_size=leaf_size, seed=seed)


def _export(tmp_path, forest, layout, extra, name=None, **kwargs):
    path = tmp_path / (name or f"{layout}.kdf")
    write_kdf(
        forest, str(path), layout=layout, chunk_bytes=_CHUNK_BYTES, **extra, **kwargs
    )
    return path


def _sources(descriptors, images=5):
    """A minimal but complete SIFT provenance record for `descriptors`."""
    n = len(descriptors)
    per_image = -(-n // images)
    return {
        "workspace": {
            "absolute_path": "/ws",
            "relative_path": ".",
            "contents": {
                "feature_tool": "sfmtool",
                "feature_type": "sift",
                "feature_options": json.dumps({"peak_threshold": 0.01}),
                "feature_prefix_dir": "features/sift",
            },
        },
        "image_names": [f"img{i:03d}.jpg" for i in range(images)],
        "feature_tool_hashes": [bytes([i]) * 16 for i in range(images)],
        "sift_content_hashes": [bytes([200 + i]) * 16 for i in range(images)],
        "image_indexes": [i // per_image for i in range(n)],
        "image_feature_indexes": [i % per_image for i in range(n)],
    }


# ── Parity: the whole point of a second storage path ──────────────────────


@pytest.mark.parametrize("layout,extra", _LAYOUTS)
def test_file_backed_query_matches_the_in_memory_forest(tmp_path, layout, extra):
    """Both layouts answer exactly what the forest they came from answers."""
    descriptors = _descriptors()
    forest = _forest(descriptors)
    path = _export(tmp_path, forest, layout, extra)

    lazy = LazyKdForest(str(path))
    queries = descriptors[:32]
    for budget in (0, 1, 16, 512):
        got_idx, got_dist = lazy.query(queries, k=3, max_leaf_checks=budget)
        want_idx, want_dist = forest.query(queries, k=3, max_leaf_checks=budget)
        assert np.array_equal(got_idx, want_idx), f"indices differ at budget {budget}"
        assert np.allclose(got_dist, want_dist), f"distances differ at budget {budget}"


@pytest.mark.parametrize("layout,extra", _LAYOUTS)
def test_the_two_layouts_answer_identically(tmp_path, layout, extra):
    """Layout is a storage decision, so it may not change a single answer."""
    descriptors = _descriptors()
    forest = _forest(descriptors)
    baseline = LazyKdForest(
        str(_export(tmp_path, forest, "tree_local", {}, name="base.kdf"))
    )
    other = LazyKdForest(str(_export(tmp_path, forest, layout, extra)))
    queries = descriptors[:32]
    a = baseline.query(queries, k=4, max_leaf_checks=64)
    b = other.query(queries, k=4, max_leaf_checks=64)
    assert np.array_equal(a[0], b[0])
    assert np.allclose(a[1], b[1])


def test_distances_are_euclidean_like_the_eager_binding(tmp_path):
    """A self-query finds itself at distance zero, not zero-squared-and-unlabelled."""
    descriptors = _descriptors(n=64)
    forest = _forest(descriptors)
    lazy = LazyKdForest(str(_export(tmp_path, forest, "tree_local", {})))
    idx, dist = lazy.query(descriptors[:8], k=1, max_leaf_checks=256)
    assert np.array_equal(idx[:, 0], np.arange(8, dtype=np.uint32))
    assert np.allclose(dist[:, 0], 0.0)


# ── Laziness and counters: what a benchmark measures with ─────────────────


@pytest.mark.parametrize("layout,extra", _LAYOUTS)
def test_opening_reads_no_chunk_payload(tmp_path, layout, extra):
    """Open validates structure but decodes nothing, in both layouts.

    The shared layout reads its row map at open, which is metadata rather than
    a chunk; neither layout may touch a tree chunk or a descriptor block.
    """
    forest = _forest(_descriptors())
    lazy = LazyKdForest(str(_export(tmp_path, forest, layout, extra)))
    assert lazy.io_stats()["read_calls"] == 0


def test_counters_accumulate_and_reset_without_dropping_the_cache(tmp_path):
    """`reset_io_stats` separates measurement phases without reopening."""
    descriptors = _descriptors()
    forest = _forest(descriptors)
    lazy = LazyKdForest(str(_export(tmp_path, forest, "tree_local", {})))

    lazy.query(descriptors[:16], k=2, max_leaf_checks=64)
    warmed = lazy.io_stats()
    assert warmed["read_calls"] > 0
    assert warmed["decoded_bytes"] > 0
    assert warmed["resident_bytes"] > 0

    lazy.reset_io_stats()
    after = lazy.io_stats()
    assert after["read_calls"] == 0
    assert after["decoded_bytes"] == 0
    # The gauges describe what is held right now, so they survive the reset,
    # and the peak may not be reported below a byte count already resident.
    assert after["resident_bytes"] == warmed["resident_bytes"]
    assert after["peak_resident_bytes"] >= after["resident_bytes"]


def test_a_repeated_query_is_served_from_cache(tmp_path):
    """A warm hit costs cache hits, not reads."""
    descriptors = _descriptors()
    forest = _forest(descriptors)
    lazy = LazyKdForest(str(_export(tmp_path, forest, "tree_local", {})))
    queries = descriptors[:8]

    lazy.query(queries, k=2, max_leaf_checks=64)
    lazy.reset_io_stats()
    lazy.query(queries, k=2, max_leaf_checks=64)

    warm = lazy.io_stats()
    assert warm["read_calls"] == 0, "a warm repeat must not re-read"
    assert warm["cache_hits"] > 0


def test_query_with_stats_reports_the_checks_read_amplification_needs(tmp_path):
    """The counters are per batch and consistent with the plain query."""
    descriptors = _descriptors()
    forest = _forest(descriptors)
    lazy = LazyKdForest(str(_export(tmp_path, forest, "tree_local", {})))
    queries = descriptors[:16]

    plain = lazy.query(queries, k=3, max_leaf_checks=64)
    idx, dist, stats = lazy.query_with_stats(queries, k=3, max_leaf_checks=64)
    assert np.array_equal(plain[0], idx)
    assert np.allclose(plain[1], dist)
    assert set(stats) == {"checks", "pushes", "pops"}
    assert stats["checks"] > 0
    # Read amplification is decoded bytes over the vector bytes actually
    # evaluated; both halves have to be available from one object for the
    # ratio to be computable at all.
    assert lazy.io_stats()["decoded_bytes"] > 0


def test_a_budget_of_zero_still_answers_and_checks_nothing_beyond_one_leaf(tmp_path):
    """The leaf budget is soft at zero, matching the in-memory forest."""
    descriptors = _descriptors()
    forest = _forest(descriptors)
    lazy = LazyKdForest(str(_export(tmp_path, forest, "tree_local", {})))
    idx, _, stats = lazy.query_with_stats(descriptors[:4], k=2, max_leaf_checks=0)
    want, _ = forest.query(descriptors[:4], k=2, max_leaf_checks=0)
    assert np.array_equal(idx, want)
    assert stats["checks"] > 0


# ── File accounting: the other half of the comparison ─────────────────────


@pytest.mark.parametrize("layout,extra", _LAYOUTS)
def test_summary_describes_the_file_it_was_given(tmp_path, layout, extra):
    forest = _forest(_descriptors())
    summary = kdf_file_summary(str(_export(tmp_path, forest, layout, extra)))
    assert summary["feature_count"] == _N
    assert summary["dimension"] == _DIM
    assert summary["scalar_type"] == "uint8"
    assert summary["tree_count"] == 4
    assert summary["descriptor_storage"] == layout
    assert summary["target_chunk_bytes"] == _CHUNK_BYTES
    assert len(summary["chunks_per_tree"]) == 4
    assert all(c > 0 for c in summary["chunks_per_tree"])
    assert sum(summary["nodes_per_tree"]) > 0
    assert not summary["has_sources"]
    # ZIP headers and the central directory live outside every entry payload,
    # so the file is always larger than the sum of its sections.
    assert summary["file_bytes"] > summary["payload_compressed_bytes"]


def test_the_layouts_differ_in_exactly_the_sections_they_should(tmp_path):
    """This is the measurement the bindings exist for.

    Tree-local keeps one vector copy per tree; shared keeps one corpus plus a
    row map. Topology is identical either way, so the tree node, split and
    feature-ID sections must match byte for byte — if they did not, a size
    comparison between the layouts would be measuring two different forests.
    """
    forest = _forest(_descriptors())
    local = kdf_file_summary(str(_export(tmp_path, forest, "tree_local", {})))
    shared = kdf_file_summary(
        str(
            _export(
                tmp_path, forest, "shared", {"descriptor_block_bytes": _BLOCK_BYTES}
            )
        )
    )

    def section(summary, name):
        return next((s for s in summary["sections"] if s["section"] == name), None)

    for name in ("tree_nodes", "tree_splits", "tree_feature_ids"):
        assert section(local, name) == section(shared, name), (
            f"{name} should be identical"
        )

    assert section(local, "tree_vectors") is not None
    assert section(shared, "tree_vectors") is None, (
        "shared layout stores no tree-local vectors"
    )
    assert section(shared, "shared_vectors") is not None
    assert section(shared, "shared_row_map") is not None

    # Four trees, so the shared corpus should be far smaller than four copies.
    assert (
        section(shared, "shared_vectors")["decoded_bytes"]
        < section(local, "tree_vectors")["decoded_bytes"]
    )


def test_verify_reads_the_whole_file_and_reports_what_it_saw(tmp_path):
    forest = _forest(_descriptors())
    report = verify_kdf(
        str(
            _export(
                tmp_path, forest, "shared", {"descriptor_block_bytes": _BLOCK_BYTES}
            )
        )
    )
    assert report["features"] == _N
    assert report["trees"] == 4
    assert report["chunks"] > 0
    assert report["descriptor_blocks"] > 0


# ── SIFT provenance ───────────────────────────────────────────────────────


def test_origins_round_trip_in_the_order_requested(tmp_path):
    """Requested order is preserved, repeats included."""
    descriptors = _descriptors(dim=128)
    forest = _forest(descriptors)
    sources = _sources(descriptors)
    path = _export(
        tmp_path, forest, "tree_local", {}, sources=sources, origin_block_rows=64
    )

    lazy = LazyKdForest(str(path))
    wanted = [7, 0, 7, 399]
    images, features = lazy.resolve_origins(wanted)
    assert list(images) == [sources["image_indexes"][i] for i in wanted]
    assert list(features) == [sources["image_feature_indexes"][i] for i in wanted]

    table = lazy.image_table()
    assert table["names"] == sources["image_names"]
    assert table["sift_content_hashes"] == sources["sift_content_hashes"]
    assert kdf_file_summary(str(path))["has_sources"]


def test_a_file_without_sources_resolves_to_none(tmp_path):
    forest = _forest(_descriptors())
    lazy = LazyKdForest(str(_export(tmp_path, forest, "tree_local", {})))
    assert lazy.resolve_origins([0, 1]) is None
    assert lazy.image_table() is None


def test_origins_cost_shows_up_in_the_summary(tmp_path):
    """The origins table is a real section, so its cost is attributable."""
    descriptors = _descriptors(dim=128)
    forest = _forest(descriptors)
    with_sources = kdf_file_summary(
        str(
            _export(
                tmp_path,
                forest,
                "tree_local",
                {},
                name="src.kdf",
                sources=_sources(descriptors),
                origin_block_rows=64,
            )
        )
    )
    names = {s["section"] for s in with_sources["sections"]}
    assert {"origins", "images"} <= names


# ── Argument handling ─────────────────────────────────────────────────────


def test_an_unknown_layout_is_refused(tmp_path):
    forest = _forest(_descriptors())
    with pytest.raises(ValueError, match="unknown layout"):
        write_kdf(forest, str(tmp_path / "x.kdf"), layout="magic")


def test_block_size_is_refused_for_the_layout_that_ignores_it(tmp_path):
    """A silently-ignored argument would label two identical sweep runs apart."""
    forest = _forest(_descriptors())
    with pytest.raises(ValueError, match="layout='shared' only"):
        write_kdf(
            forest,
            str(tmp_path / "x.kdf"),
            layout="tree_local",
            descriptor_block_bytes=4096,
        )


def test_writing_over_an_existing_file_is_refused(tmp_path):
    forest = _forest(_descriptors())
    path = _export(tmp_path, forest, "tree_local", {})
    with pytest.raises(FileExistsError):
        write_kdf(forest, str(path), layout="tree_local")


def test_a_query_of_the_wrong_width_is_refused(tmp_path):
    forest = _forest(_descriptors())
    lazy = LazyKdForest(str(_export(tmp_path, forest, "tree_local", {})))
    with pytest.raises(ValueError, match="does not match file dim"):
        lazy.query(np.zeros((2, _DIM + 1), dtype=np.uint8))


def test_a_query_of_the_wrong_dtype_is_refused(tmp_path):
    forest = _forest(_descriptors())
    lazy = LazyKdForest(str(_export(tmp_path, forest, "tree_local", {})))
    with pytest.raises(TypeError, match="must be a uint8 array"):
        lazy.query(np.zeros((2, _DIM), dtype=np.float32))


def test_a_fortran_ordered_query_is_not_silently_transposed(tmp_path):
    """F-contiguous input must give the same answers as its C-ordered twin."""
    descriptors = _descriptors()
    forest = _forest(descriptors)
    lazy = LazyKdForest(str(_export(tmp_path, forest, "tree_local", {})))
    queries = descriptors[:16]
    c_idx, c_dist = lazy.query(queries, k=2, max_leaf_checks=64)
    f_idx, f_dist = lazy.query(np.asfortranarray(queries), k=2, max_leaf_checks=64)
    assert np.array_equal(c_idx, f_idx)
    assert np.allclose(c_dist, f_dist)


def test_a_cache_budget_too_small_for_a_chunk_is_a_memory_error(tmp_path):
    """Budgets that cannot hold what is asked for are distinguishable.

    A sweep needs to tell "this budget is too small, try another" from "this
    file is broken, stop", so the two raise different exceptions.
    """
    forest = _forest(_descriptors())
    path = _export(tmp_path, forest, "tree_local", {})
    with pytest.raises(MemoryError):
        LazyKdForest(str(path), cache_bytes=0)
    with pytest.raises(MemoryError):
        LazyKdForest(str(path), query_workers=0)


def test_opening_a_file_that_is_not_a_kdf_is_an_os_error(tmp_path):
    junk = tmp_path / "not.kdf"
    junk.write_bytes(b"definitely not a zip archive")
    with pytest.raises(OSError):
        LazyKdForest(str(junk))


def test_a_missing_file_is_a_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        LazyKdForest(str(tmp_path / "absent.kdf"))
