# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the measurement item bindings: putting a stored measurement on
the bench, reading it back, and the commit that writes it into the
reconstruction."""

import numpy as np
import pytest

from sfmtool import bench as bench_module
from sfmtool.bench import (
    Bench,
    commit_measurement,
    parse_stated_distance,
    put_measurement,
    set_measurement_endpoint,
    set_measurement_label,
    set_stated_distance,
    start_measurement,
)
from sfmtool.fileio import LENGTH_UNIT_NAMES
from sfmtool.reconstruction import EditedReconstruction, SfmrReconstruction
from tests.conftest import SEOUL_BULL_GROUND_TRUTH


@pytest.fixture(scope="module")
def measured():
    """The seoul_bull ground truth, read and never written, with two distance
    measurements."""
    recon = SfmrReconstruction.load(SEOUL_BULL_GROUND_TRUTH)
    distance = {
        "distances": np.array([0.42, 12.5], dtype=np.float32),
        "units": np.array(
            [LENGTH_UNIT_NAMES.index("m"), LENGTH_UNIT_NAMES.index("cm")],
            dtype=np.uint8,
        ),
        "image_indexes": np.array([[0, 3], [2, 2]], dtype=np.uint32),
        "endpoints_xy": np.array(
            [[[10.5, 20.25], [100.0, 200.5]], [[120.0, 220.0], [200.5, 350.25]]],
            dtype=np.float32,
        ),
        "labels": ["ruler edge", ""],
    }
    return recon.clone_with_changes(measurements={"distance": distance})


@pytest.fixture
def edited(measured):
    return EditedReconstruction(measured)


def test_a_put_copies_the_row_and_labels_the_item(edited):
    bench = Bench()
    nxt, item = put_measurement(bench, edited, "distance", 0, version=7)
    assert len(bench) == 0
    assert nxt.labels == ["distance #0"]
    assert nxt.kind("distance #0") == "measurement"
    assert nxt.track("distance #0") is None
    assert item.kind == "distance"
    assert item.label == "ruler edge"
    assert item.origin == {"version": 7, "kind": "distance", "index": 0}
    assert item.endpoints == [(0, (10.5, 20.25)), (3, (100.0, 200.5))]
    distance, unit = item.distance
    assert unit == "m"
    assert distance == pytest.approx(0.42, rel=1e-6)
    assert nxt.measurement("distance #0").label == "ruler edge"
    again, _ = put_measurement(nxt, edited, "distance", 0)
    assert again.labels == ["distance #0", "distance #0 (2)"]


def test_a_row_or_kind_that_is_not_there_is_refused(edited):
    with pytest.raises(ValueError, match="there is no distance #2"):
        put_measurement(Bench(), edited, "distance", 2)
    with pytest.raises(ValueError, match="not a kind of measurement"):
        put_measurement(Bench(), edited, "gps", 0)


def test_a_commit_of_the_row_as_it_stands_writes_nothing(edited):
    _, item = put_measurement(Bench(), edited, "distance", 1)
    nxt, report = commit_measurement(edited, item, 1, node="bull")
    assert report["wrote"] is False
    assert report["index"] == 1
    assert report["label"] == (
        "Committed distance #1 in bull: no effect, the row already holds it"
    )


def test_a_commit_with_no_row_appends_one_and_keeps_the_points(edited):
    _, item = put_measurement(Bench(), edited, "distance", 0)
    nxt, report = commit_measurement(edited, item, None, node="bull")
    assert report == {
        "index": 2,
        "replaced": False,
        "wrote": True,
        "label": "Added distance #2 in bull",
    }
    assert nxt.point_count == edited.point_count
    plain = nxt.materialize()[0]
    assert plain.distance_measurement_count == 3
    assert plain.measurements["distance"]["labels"] == ["ruler edge", "", "ruler edge"]
    # The value committed into is unchanged.
    assert edited.materialize()[0].distance_measurement_count == 2


def test_a_commit_past_the_rows_is_refused(edited):
    _, item = put_measurement(Bench(), edited, "distance", 0)
    with pytest.raises(ValueError, match="there is no distance #5"):
        commit_measurement(edited, item, 5)


def test_replace_takes_a_measurement_and_with_origin_reseats_it(edited):
    bench, item = put_measurement(Bench(), edited, "distance", 0)
    seated = item.with_origin(9, 1)
    assert seated.origin == {"version": 9, "kind": "distance", "index": 1}
    assert item.origin["index"] == 0
    nxt = bench.replace("distance #0", seated)
    assert nxt.measurement("distance #0").origin["index"] == 1
    assert bench.measurement("distance #0").origin["index"] == 0
    with pytest.raises(ValueError, match="EditableTrack or an EditableMeasurement"):
        bench.replace("distance #0", 3)
    assert isinstance(item, bench_module.EditableMeasurement)


def test_setting_the_label_gives_a_new_item_and_commits(edited):
    _, item = put_measurement(Bench(), edited, "distance", 0)
    renamed = set_measurement_label(item, "tape across the base")
    assert renamed.label == "tape across the base"
    assert item.label == "ruler edge"
    assert renamed.endpoints == item.endpoints
    assert renamed.distance == item.distance
    assert renamed.origin == item.origin
    nxt, report = commit_measurement(edited, renamed, 0)
    assert report["wrote"] is True
    labels = nxt.materialize()[0].measurements["distance"]["labels"]
    assert labels == ["tape across the base", ""]


def test_setting_the_stated_distance_gives_a_new_item_or_refuses(edited):
    _, item = put_measurement(Bench(), edited, "distance", 0)
    restated = set_stated_distance(item, 4.2, "cm")
    assert restated.distance == (4.2, "cm")
    assert item.distance[1] == "m"
    assert restated.endpoints == item.endpoints
    with pytest.raises(ValueError, match="must be finite and strictly positive"):
        set_stated_distance(item, 0.0, "cm")
    with pytest.raises(ValueError, match="is not a unit"):
        set_stated_distance(item, 1.0, "yd")


def test_a_typed_distance_reads_its_unit_or_the_bare_one():
    assert parse_stated_distance("12in") == (12.0, "in")
    assert parse_stated_distance("30.48 cm", "m") == (30.48, "cm")
    assert parse_stated_distance("304.8", "mm") == (304.8, "mm")
    assert parse_stated_distance("2") == (2.0, "m")
    with pytest.raises(ValueError, match="is not a distance"):
        parse_stated_distance("twelve")
    # Text holding a multi-byte character is refused, not cut inside one.
    for text in ["12é", "é", "12mé", "١٢"]:
        with pytest.raises(ValueError, match="is not a distance"):
            parse_stated_distance(text)
    assert parse_stated_distance("12 cm") == (12.0, "cm")
    with pytest.raises(ValueError, match='"yd" is not a unit'):
        parse_stated_distance("3yd")
    with pytest.raises(ValueError, match="must be finite and strictly positive"):
        parse_stated_distance("-1cm")


def test_setting_an_endpoint_gives_a_new_item_or_refuses(edited):
    _, item = put_measurement(Bench(), edited, "distance", 0)
    moved = set_measurement_endpoint(edited, item, 1, 5, (12.5, 40.0))
    assert moved.endpoints == [(0, (10.5, 20.25)), (5, (12.5, 40.0))]
    assert item.endpoints[1] == (3, (100.0, 200.5))
    assert moved.distance == item.distance
    assert moved.label == item.label
    nxt, report = commit_measurement(edited, moved, 0)
    assert report["wrote"] is True
    stored = nxt.materialize()[0].measurements["distance"]
    assert stored["image_indexes"][0].tolist() == [0, 5]
    with pytest.raises(ValueError, match=r"endpoint B's pixel \(-1, 3\) is outside"):
        set_measurement_endpoint(edited, item, 1, 5, (-1.0, 3.0))
    with pytest.raises(ValueError, match="past the reconstruction's"):
        set_measurement_endpoint(edited, item, 0, 999, (1.0, 1.0))
    with pytest.raises(ValueError, match="there is no endpoint 2"):
        set_measurement_endpoint(edited, item, 2, 0, (1.0, 1.0))


def test_a_measurement_started_from_a_pixel_holds_endpoint_a_alone(measured, edited):
    bench, item = start_measurement(Bench(), edited, 3, (141.6, 198.4))
    stem = measured.image_names[3].rsplit(".", 1)[0]
    assert bench.labels == [f"distance from {stem}@142,198"]
    assert bench.kind(bench.labels[0]) == "measurement"
    assert item.origin is None
    assert item.label == ""
    assert item.distance is None
    assert item.endpoints[0] == (3, pytest.approx((141.6, 198.4)))
    assert item.endpoints[1] is None
    again, _ = start_measurement(bench, edited, 3, (142.4, 197.6))
    assert again.labels[1] == f"distance from {stem}@142,198 (2)"
    with pytest.raises(ValueError, match="is outside image 3"):
        start_measurement(Bench(), edited, 3, (-1.0, 5.0))
