# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the persistent `.kdf` forest bindings.

These bindings expose the persistent format to Python. The tests confirm that
file-backed answers match the in-memory forest, that the single descriptor
corpus remains lazy, and that its counters and SIFT provenance mean what they
say.
"""

import json
import shutil

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool._sfmtool.spatial import (
    KdForest,
    LazyKdForest,
    kdf_file_summary,
    verify_kdf,
    verify_sift_sources,
    write_kdf,
)
from sfmtool.cli import main
from sfmtool.sift.file import SiftReader, get_sift_path_for_image

# Small enough to stay fast, wide enough that a few-hundred-byte chunk target
# still splits each tree into many chunks — a traversal that never crosses a
# chunk boundary would exercise none of the lazy path.
_N = 400
_DIM = 32
_CHUNK_BYTES = 4096
_BLOCK_BYTES = 1024

_BLOCK_CASES = [
    pytest.param("corpus", {"descriptor_block_bytes": _BLOCK_BYTES}, id="corpus")
]


def _descriptors(n=_N, dim=_DIM, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, dim), dtype=np.uint8)


def _forest(descriptors, num_trees=4, leaf_size=8, seed=7):
    return KdForest(descriptors, num_trees=num_trees, leaf_size=leaf_size, seed=seed)


def _export(tmp_path, forest, label, extra, name=None, **kwargs):
    path = tmp_path / (name or f"{label}.kdf")
    write_kdf(forest, str(path), chunk_bytes=_CHUNK_BYTES, **extra, **kwargs)
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
        "positions": np.arange(n * 2, dtype=np.float32).reshape(n, 2),
        "affine_shapes": np.broadcast_to(np.eye(2, dtype=np.float32), (n, 2, 2)).copy(),
    }


# ── Parity: the whole point of a second storage path ──────────────────────


@pytest.mark.parametrize("label,extra", _BLOCK_CASES)
def test_file_backed_query_matches_the_in_memory_forest(tmp_path, label, extra):
    """The persistent forest answers exactly what its source forest answers."""
    descriptors = _descriptors()
    forest = _forest(descriptors)
    path = _export(tmp_path, forest, label, extra)

    lazy = LazyKdForest(str(path))
    queries = descriptors[:32]
    for budget in (0, 1, 16, 512):
        got_idx, got_dist = lazy.query(queries, k=3, max_leaf_checks=budget)
        want_idx, want_dist = forest.query(queries, k=3, max_leaf_checks=budget)
        assert np.array_equal(got_idx, want_idx), f"indices differ at budget {budget}"
        assert np.allclose(got_dist, want_dist), f"distances differ at budget {budget}"


@pytest.mark.parametrize("label,extra", _BLOCK_CASES)
def test_descriptor_block_size_changes_no_answer(tmp_path, label, extra):
    """Descriptor blocking is a storage decision, not a query decision."""
    descriptors = _descriptors()
    forest = _forest(descriptors)
    baseline = LazyKdForest(
        str(_export(tmp_path, forest, "corpus", {}, name="base.kdf"))
    )
    other = LazyKdForest(str(_export(tmp_path, forest, label, extra)))
    queries = descriptors[:32]
    a = baseline.query(queries, k=4, max_leaf_checks=64)
    b = other.query(queries, k=4, max_leaf_checks=64)
    assert np.array_equal(a[0], b[0])
    assert np.allclose(a[1], b[1])


def test_distances_are_euclidean_like_the_eager_binding(tmp_path):
    """A self-query finds itself at distance zero, not zero-squared-and-unlabelled."""
    descriptors = _descriptors(n=64)
    forest = _forest(descriptors)
    lazy = LazyKdForest(str(_export(tmp_path, forest, "corpus", {})))
    idx, dist = lazy.query(descriptors[:8], k=1, max_leaf_checks=256)
    assert np.array_equal(idx[:, 0], np.arange(8, dtype=np.uint32))
    assert np.allclose(dist[:, 0], 0.0)


# ── Laziness and counters: what a benchmark measures with ─────────────────


@pytest.mark.parametrize("label,extra", _BLOCK_CASES)
def test_opening_reads_no_chunk_payload(tmp_path, label, extra):
    """Open validates structure but decodes no tree or corpus frame.

    The row map is addressing metadata rather than a cache payload; open may not
    touch a tree chunk, descriptor block, or geometry block.
    """
    forest = _forest(_descriptors())
    lazy = LazyKdForest(str(_export(tmp_path, forest, label, extra)))
    assert lazy.io_stats()["read_calls"] == 0


def test_counters_accumulate_and_reset_without_dropping_the_cache(tmp_path):
    """`reset_io_stats` separates measurement phases without reopening."""
    descriptors = _descriptors()
    forest = _forest(descriptors)
    lazy = LazyKdForest(str(_export(tmp_path, forest, "corpus", {})))

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
    lazy = LazyKdForest(str(_export(tmp_path, forest, "corpus", {})))
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
    lazy = LazyKdForest(str(_export(tmp_path, forest, "corpus", {})))
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
    lazy = LazyKdForest(str(_export(tmp_path, forest, "corpus", {})))
    idx, _, stats = lazy.query_with_stats(descriptors[:4], k=2, max_leaf_checks=0)
    want, _ = forest.query(descriptors[:4], k=2, max_leaf_checks=0)
    assert np.array_equal(idx, want)
    assert stats["checks"] > 0


# ── File accounting: the other half of the comparison ─────────────────────


def test_default_export_uses_two_kib_descriptor_blocks(tmp_path):
    descriptors = _descriptors(n=64, dim=128)
    forest = KdForest(descriptors)
    path = tmp_path / "default.kdf"
    write_kdf(forest, str(path))

    summary = kdf_file_summary(str(path))
    assert summary["descriptor_block_rows"] == 16
    assert summary["feature_count"] == 64


@pytest.mark.parametrize("label,extra", _BLOCK_CASES)
def test_summary_describes_the_file_it_was_given(tmp_path, label, extra):
    forest = _forest(_descriptors())
    summary = kdf_file_summary(str(_export(tmp_path, forest, label, extra)))
    assert summary["feature_count"] == _N
    assert summary["dimension"] == _DIM
    assert summary["scalar_type"] == "uint8"
    assert summary["tree_count"] == 4
    assert summary["descriptor_block_rows"] == _BLOCK_BYTES // _DIM
    assert summary["target_chunk_bytes"] == _CHUNK_BYTES
    assert len(summary["chunks_per_tree"]) == 4
    assert all(c > 0 for c in summary["chunks_per_tree"])
    assert sum(summary["nodes_per_tree"]) > 0
    assert not summary["has_sources"]
    # ZIP headers and the central directory live outside every entry payload,
    # so the file is always larger than the sum of its sections.
    assert summary["file_bytes"] > summary["payload_compressed_bytes"]


def test_descriptors_are_stored_once_outside_the_trees(tmp_path):
    descriptors = _descriptors()
    forest = _forest(descriptors, num_trees=4)
    summary = kdf_file_summary(
        str(
            _export(
                tmp_path, forest, "corpus", {"descriptor_block_bytes": _BLOCK_BYTES}
            )
        )
    )

    def section(name):
        return next((s for s in summary["sections"] if s["section"] == name), None)

    assert section("descriptors")["decoded_bytes"] == _N * _DIM
    assert section("storage_row_map")["decoded_bytes"] == _N * 4
    assert section("descriptor_block_offsets") is not None
    assert section("tree_vectors") is None


def test_summary_reports_real_decoded_sizes_not_the_stored_frame(tmp_path):
    """Decoded bytes must be the uncompressed length, not the frame length.

    `.kdf` entries are ZIP STORE wrapping a zstd frame, so the ZIP directory's
    "uncompressed" size equals the compressed size. A summary built on it reports
    every section at exactly 100% — a plausible-looking number rather than an
    obvious failure. Compressible descriptors make the difference unmistakable.
    """
    # All-zero descriptors compress to almost nothing, so a decoded size that
    # merely echoed the stored frame would be off by orders of magnitude.
    descriptors = np.zeros((_N, _DIM), dtype=np.uint8)
    descriptors[:, 0] = np.arange(_N, dtype=np.uint8)  # keep the tree splittable
    forest = _forest(descriptors)
    summary = kdf_file_summary(str(_export(tmp_path, forest, "corpus", {})))

    vectors = next(s for s in summary["sections"] if s["section"] == "descriptors")
    assert vectors["decoded_bytes"] == _N * _DIM
    assert vectors["compressed_bytes"] < vectors["decoded_bytes"] // 10

    # And the whole payload compresses, rather than reporting a 1.0 ratio.
    assert summary["payload_compressed_bytes"] < summary["payload_decoded_bytes"]
    assert (
        sum(s["decoded_bytes"] for s in summary["sections"])
        == (summary["payload_decoded_bytes"])
    )


def test_verify_reads_the_whole_file_and_reports_what_it_saw(tmp_path):
    forest = _forest(_descriptors())
    report = verify_kdf(
        str(
            _export(
                tmp_path, forest, "corpus", {"descriptor_block_bytes": _BLOCK_BYTES}
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
        tmp_path, forest, "corpus", {}, sources=sources, origin_block_rows=64
    )

    lazy = LazyKdForest(str(path))
    wanted = [7, 0, 7, 399]
    images, features = lazy.resolve_origins(wanted)
    assert list(images) == [sources["image_indexes"][i] for i in wanted]
    assert list(features) == [sources["image_feature_indexes"][i] for i in wanted]
    geometry = lazy.resolve_feature_geometry(wanted)
    expected = np.concatenate(
        [sources["positions"][:, None, :], sources["affine_shapes"]], axis=1
    )
    np.testing.assert_array_equal(geometry, expected[wanted])

    table = lazy.image_table()
    assert table["names"] == sources["image_names"]
    assert table["sift_content_hashes"] == sources["sift_content_hashes"]
    assert kdf_file_summary(str(path))["has_sources"]
    assert kdf_file_summary(str(path))["has_feature_geometry"]


def test_a_file_without_sources_resolves_to_none(tmp_path):
    forest = _forest(_descriptors())
    lazy = LazyKdForest(str(_export(tmp_path, forest, "corpus", {})))
    assert lazy.resolve_origins([0, 1]) is None
    assert lazy.resolve_feature_geometry([0, 1]) is None
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
                "corpus",
                {},
                name="src.kdf",
                sources=_sources(descriptors),
                origin_block_rows=64,
            )
        )
    )
    names = {s["section"] for s in with_sources["sections"]}
    assert {
        "origins",
        "images",
        "feature_geometry",
        "geometry_block_offsets",
    } <= names


def test_source_verifier_uses_the_extracted_seoul_bull_sift(
    isolated_seoul_bull_image, tmp_path
):
    """One real extraction covers success, relocation, and every source failure.

    The included 270x480 Seoul Bull image keeps this integration check small.
    All KDF cases below reuse its one generated `.sift` file, so adding the
    source-verification surface does not multiply extraction cost.
    """
    workspace = isolated_seoul_bull_image.parent
    runner = CliRunner()
    result = runner.invoke(
        main, ["ws", "init", "--feature-tool", "sfmtool", str(workspace)]
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(main, ["sift", "--extract", str(isolated_seoul_bull_image)])
    assert result.exit_code == 0, result.output

    config = json.loads((workspace / ".sfm-workspace.json").read_text())
    sift_path = get_sift_path_for_image(isolated_seoul_bull_image)
    sift = SiftReader(sift_path)
    all_descriptors = sift.read_descriptors()
    all_positions, all_affine_shapes = sift.read_positions_and_shapes()
    from scripts.benchmark_kdf_layouts import load_corpus

    (
        benchmark_descriptors,
        benchmark_names,
        _,
        _,
        benchmark_positions,
        benchmark_shapes,
    ) = load_corpus(sift_path.parent)
    assert benchmark_names == [isolated_seoul_bull_image.name]
    np.testing.assert_array_equal(benchmark_descriptors, all_descriptors)
    np.testing.assert_array_equal(benchmark_positions, all_positions)
    np.testing.assert_array_equal(benchmark_shapes, all_affine_shapes)
    feature_count = min(64, len(all_descriptors))
    assert feature_count > 1
    descriptors = np.array(all_descriptors[:feature_count], copy=True)

    sources = {
        "workspace": {
            "absolute_path": str(workspace),
            "relative_path": ".",
            "contents": {
                "feature_tool": config["feature_tool"],
                "feature_type": config["feature_type"],
                "feature_options": json.dumps(config["feature_options"]),
                "feature_prefix_dir": config["feature_prefix_dir"],
            },
        },
        "image_names": [isolated_seoul_bull_image.name],
        "feature_tool_hashes": [
            bytes.fromhex(sift.content_hash["feature_tool_xxh128"])
        ],
        "sift_content_hashes": [bytes.fromhex(sift.content_hash["content_xxh128"])],
        "image_indexes": [0] * feature_count,
        "image_feature_indexes": list(range(feature_count)),
        "positions": np.array(all_positions[:feature_count], copy=True),
        "affine_shapes": np.array(all_affine_shapes[:feature_count], copy=True),
    }

    def export(name, values, provenance):
        return _export(
            workspace,
            _forest(values, num_trees=2),
            "corpus",
            {"descriptor_block_bytes": 4096},
            name=name,
            sources=provenance,
            origin_block_rows=32,
        )

    valid = export("valid.kdf", descriptors, sources)

    wrong_hash_sources = dict(sources)
    wrong_hash_sources["sift_content_hashes"] = [bytes(16)]
    wrong_hash = export("wrong-hash.kdf", descriptors, wrong_hash_sources)

    out_of_bounds_sources = dict(sources)
    out_of_bounds_sources["image_feature_indexes"] = list(range(feature_count))
    out_of_bounds_sources["image_feature_indexes"][-1] = len(all_descriptors)
    out_of_bounds = export("out-of-bounds.kdf", descriptors, out_of_bounds_sources)

    changed = descriptors.copy()
    changed[0, 0] ^= 1
    descriptor_mismatch = export("descriptor-mismatch.kdf", changed, sources)

    geometry_mismatch_sources = dict(sources)
    geometry_mismatch_sources["positions"] = sources["positions"].copy()
    geometry_mismatch_sources["positions"][0, 0] += 1.0
    geometry_mismatch = export(
        "geometry-mismatch.kdf", descriptors, geometry_mismatch_sources
    )

    # Move the whole workspace after export. The relative location beside each
    # KDF must win over the now-stale recorded absolute path.
    relocated = tmp_path / "relocated-workspace"
    workspace.rename(relocated)
    valid = relocated / valid.name
    wrong_hash = relocated / wrong_hash.name
    out_of_bounds = relocated / out_of_bounds.name
    descriptor_mismatch = relocated / descriptor_mismatch.name
    geometry_mismatch = relocated / geometry_mismatch.name
    sift_path = relocated / sift_path.relative_to(workspace)

    report = verify_sift_sources(str(valid))
    assert report["features"] == feature_count
    geometry = LazyKdForest(str(valid)).resolve_feature_geometry([3, 0, 3])
    np.testing.assert_array_equal(geometry[:, 0], sources["positions"][[3, 0, 3]])
    np.testing.assert_array_equal(geometry[:, 1:], sources["affine_shapes"][[3, 0, 3]])

    with pytest.raises(OSError, match="SIFT identity mismatch"):
        verify_sift_sources(str(wrong_hash))
    with pytest.raises(OSError, match="image_feature_index out of range"):
        verify_sift_sources(str(out_of_bounds))
    with pytest.raises(OSError, match="source descriptor differs"):
        verify_sift_sources(str(descriptor_mismatch))
    with pytest.raises(OSError, match="source feature geometry differs"):
        verify_sift_sources(str(geometry_mismatch))

    missing = sift_path.with_suffix(sift_path.suffix + ".missing")
    shutil.move(sift_path, missing)
    with pytest.raises(FileNotFoundError, match="SIFT source is missing"):
        verify_sift_sources(str(valid))

    # Provenance is optional audit data: embedded descriptors and origins stay
    # usable even while their source archive is absent.
    lazy = LazyKdForest(str(valid))
    images, features = lazy.resolve_origins([0])
    assert list(images) == [0]
    assert list(features) == [0]
    indexes, distances = lazy.query(descriptors[:1], k=1, max_leaf_checks=64)
    assert indexes.shape == distances.shape == (1, 1)


# ── Argument handling ─────────────────────────────────────────────────────


def test_the_removed_layout_keyword_is_refused(tmp_path):
    forest = _forest(_descriptors())
    with pytest.raises(TypeError, match="layout"):
        write_kdf(forest, str(tmp_path / "x.kdf"), layout="magic")


def test_descriptor_block_size_must_be_positive(tmp_path):
    forest = _forest(_descriptors())
    with pytest.raises(ValueError, match="must be positive"):
        write_kdf(
            forest,
            str(tmp_path / "x.kdf"),
            descriptor_block_bytes=0,
        )


def test_writing_over_an_existing_file_is_refused(tmp_path):
    forest = _forest(_descriptors())
    path = _export(tmp_path, forest, "corpus", {})
    with pytest.raises(FileExistsError):
        write_kdf(forest, str(path))


def test_a_query_of_the_wrong_width_is_refused(tmp_path):
    forest = _forest(_descriptors())
    lazy = LazyKdForest(str(_export(tmp_path, forest, "corpus", {})))
    with pytest.raises(ValueError, match="does not match file dim"):
        lazy.query(np.zeros((2, _DIM + 1), dtype=np.uint8))


def test_a_query_of_the_wrong_dtype_is_refused(tmp_path):
    forest = _forest(_descriptors())
    lazy = LazyKdForest(str(_export(tmp_path, forest, "corpus", {})))
    with pytest.raises(TypeError, match="must be a uint8 array"):
        lazy.query(np.zeros((2, _DIM), dtype=np.float32))


def test_a_fortran_ordered_query_is_not_silently_transposed(tmp_path):
    """F-contiguous input must give the same answers as its C-ordered twin."""
    descriptors = _descriptors()
    forest = _forest(descriptors)
    lazy = LazyKdForest(str(_export(tmp_path, forest, "corpus", {})))
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
    path = _export(tmp_path, forest, "corpus", {})
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


# ── Descriptor ordering ───────────────────────────────────────────────────


def test_leaf_layout_describes_every_leaf(tmp_path):
    """The leaf hypergraph an ordering policy is optimized against."""
    descriptors = _descriptors()
    forest = _forest(descriptors, num_trees=3, leaf_size=8)
    assert forest.num_trees == 3
    seen = set()
    for tree in range(forest.num_trees):
        ids, starts = forest.leaf_layout(tree)
        # Every point appears exactly once in a tree's leaf order.
        assert sorted(ids.tolist()) == list(range(_N))
        assert starts[0] == 0
        assert list(starts) == sorted(starts)
        assert starts[-1] < len(ids)
        # Leaves partition the ids, and none is empty.
        bounds = list(starts) + [len(ids)]
        assert all(b > a for a, b in zip(bounds, bounds[1:]))
        seen.add(tuple(ids.tolist()))
    # Randomized trees give different orders, which is why one tree's order
    # cannot serve the others.
    assert len(seen) == 3


def test_leaf_layout_rejects_a_tree_that_does_not_exist(tmp_path):
    forest = _forest(_descriptors(), num_trees=2)
    with pytest.raises(IndexError, match="out of range"):
        forest.leaf_layout(2)


def test_an_explicit_descriptor_order_changes_no_answer(tmp_path):
    """Reordering the corpus is invisible above the storage layer.

    This is what makes an ordering policy safe to try: the stored row map is what
    a reader follows, so a different order is a different file with identical
    results.
    """
    descriptors = _descriptors()
    forest = _forest(descriptors)
    queries = descriptors[:16]

    default = LazyKdForest(
        str(
            _export(
                tmp_path, forest, "corpus", {"descriptor_block_bytes": _BLOCK_BYTES}
            )
        )
    )
    want = default.query(queries, k=3, max_leaf_checks=64)

    rng = np.random.default_rng(1)
    shuffled = rng.permutation(_N).astype(np.uint32).tolist()
    path = tmp_path / "reordered.kdf"
    write_kdf(
        forest,
        str(path),
        chunk_bytes=_CHUNK_BYTES,
        descriptor_block_bytes=_BLOCK_BYTES,
        descriptor_order=shuffled,
    )
    got = LazyKdForest(str(path)).query(queries, k=3, max_leaf_checks=64)
    assert np.array_equal(want[0], got[0])
    assert np.allclose(want[1], got[1])
    assert verify_kdf(str(path))["features"] == _N


def test_a_malformed_descriptor_order_is_refused(tmp_path):
    forest = _forest(_descriptors())
    for order, match in [
        (list(range(_N - 1)), "expected"),
        ([0] * _N, "repeats"),
    ]:
        with pytest.raises((ValueError, OSError)):
            write_kdf(
                forest,
                str(tmp_path / f"bad{len(order)}{order[0]}.kdf"),
                descriptor_block_bytes=_BLOCK_BYTES,
                descriptor_order=order,
            )


def test_kdf_matcher_validates_before_self_join(tmp_path):
    from sfmtool._sfmtool.matching import background_floor_clusters_kdf

    desc = _descriptors(n=16)
    path = _export(tmp_path, _forest(desc), "corpus", {"descriptor_block_bytes": 128})
    for d, starts in [(2**64 - 1, [0, 16]), (2, [0, 17]), (2, [1, 16])]:
        with pytest.raises(ValueError):
            background_floor_clusters_kdf(
                str(path), np.array(starts, dtype=np.uint32), d=d
            )
    with pytest.raises(FileNotFoundError):
        background_floor_clusters_kdf(
            str(tmp_path / "missing.kdf"), np.array([0, 16], dtype=np.uint32)
        )


def test_reloaded_forest_preserves_default_check_budget(tmp_path):
    from sfmtool._sfmtool.spatial import read_kdf

    desc = _descriptors()
    forest = KdForest(desc, num_trees=4, max_leaf_checks=32)
    path = _export(tmp_path, forest, "corpus", {"descriptor_block_bytes": 128})
    loaded = read_kdf(str(path))
    assert loaded.max_leaf_checks == 32
    for actual, expected in zip(
        loaded.query(desc[:10]), forest.query(desc[:10]), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
