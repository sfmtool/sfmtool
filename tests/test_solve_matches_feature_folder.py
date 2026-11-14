# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""A `.matches` file is used with the `.sift` files it was matched on.

The workspace here is matched once, then its feature definition is changed
(`sfm ws init --force` with another grey conversion, as `sfm ws upgrade`
changes it) and its `.sift` files extracted again into the new folder, where
each image has a different number of features. A solve, or `sfm to-colmap-db`,
from the earlier `.matches` file must read the earlier folder, and refuse
before it builds anything when that folder's files are gone or changed.
"""

import shutil
from pathlib import Path

import numpy as np
import pycolmap
import pytest
from click.testing import CliRunner

import sfmtool.fileio
from sfmtool import load_workspace_config
from sfmtool.cli import main
from sfmtool.colmap.convention import pose_canonical_to_colmap
from sfmtool.colmap.io import save_colmap_binary
from sfmtool.fileio import SiftFile, read_colmap_binary, read_matches
from sfmtool.merge.reconstructions import merge_reconstructions
from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.visualization._epipolar_display import _require_sift_path

from .conftest import SEOUL_BULL_DIR, SEOUL_BULL_GROUND_TRUTH


def _invoke(*args):
    return CliRunner().invoke(main, [str(a) for a in args])


@pytest.fixture(scope="module")
def moved_features_workspace(tmp_path_factory) -> tuple[Path, Path, str, str]:
    """A workspace with a `.matches` file made on its first feature folder,
    and its `.sift` files extracted again into a second one.

    Returns `(workspace, matches_path, old_folder, new_folder)`. Tests copy it.
    """
    workspace = tmp_path_factory.mktemp("moved_features") / "ws"
    workspace.mkdir()
    # Eight of the 17 images: enough verified pairs to triangulate points in
    # every image, and half the extraction and matching of all of them.
    for image in sorted(SEOUL_BULL_DIR.glob("seoul_bull_sculpture_*.jpg"))[:8]:
        shutil.copy(image, workspace / image.name)
    shutil.copy(SEOUL_BULL_DIR / "camera_config.json", workspace)

    result = _invoke("ws", "init", workspace)
    assert result.exit_code == 0, result.output
    old_folder = load_workspace_config(workspace)["feature_prefix_dir"]
    matches_path = workspace / "old.matches"
    result = _invoke("match", "--exhaustive", "--output", matches_path, workspace)
    assert result.exit_code == 0, result.output

    result = _invoke("ws", "init", "--force", "--image-to-gray", "G", workspace)
    assert result.exit_code == 0, result.output
    new_folder = load_workspace_config(workspace)["feature_prefix_dir"]
    assert new_folder != old_folder
    result = _invoke("sift", "--extract", workspace)
    assert result.exit_code == 0, result.output
    return workspace, matches_path, old_folder, new_folder


@pytest.fixture
def workspace_copy(moved_features_workspace, tmp_path):
    workspace, matches_path, old_folder, new_folder = moved_features_workspace
    copy = tmp_path / "ws"
    shutil.copytree(workspace, copy)
    return copy, copy / matches_path.name, old_folder, new_folder


def _positions(sift_path: Path) -> np.ndarray:
    with SiftFile(sift_path) as reader:
        return np.asarray(reader.read_positions(), dtype=np.float32)


def _map_at_true_poses(database_path, image_path, output_path, options=None):
    """Stand in for `pycolmap.global_mapping`, deterministically.

    Every image of the database the solve wrote is registered at its
    seoul_bull ground-truth pose and camera, matched by name. The verified
    inlier matches are linked into tracks, a track that holds two features of
    one image is dropped, and each remaining track is placed at the midpoint
    of its observations' rays, through the database's keypoints. Each
    observation keeps its database feature index. Like the real call, it
    writes model 0 under `output_path` and returns `{0: reconstruction}`.
    """
    truth = SfmrReconstruction.load(SEOUL_BULL_GROUND_TRUTH)
    (truth_camera,) = truth.cameras
    params = truth_camera.to_dict()["parameters"]
    quats, trans = pose_canonical_to_colmap(
        np.asarray(truth.quaternions_wxyz), np.asarray(truth.translations)
    )
    truth_index = {name: i for i, name in enumerate(truth.image_names)}

    with pycolmap.Database.open(str(database_path)) as db:
        db_cameras = db.read_all_cameras()
        db_images = sorted(db.read_all_images(), key=lambda im: im.image_id)
        keypoints = {
            im.image_id: np.asarray(db.read_keypoints(im.image_id))[:, :2]
            for im in db_images
        }
        pair_ids, geometries = db.read_two_view_geometries()

    reconstruction = pycolmap.Reconstruction()
    for camera in db_cameras:
        assert (camera.width, camera.height) == (
            truth_camera.width,
            truth_camera.height,
        )
        reconstruction.add_camera_with_trivial_rig(
            pycolmap.Camera(
                model="SIMPLE_RADIAL",
                width=truth_camera.width,
                height=truth_camera.height,
                params=[
                    params["focal_length"],
                    params["principal_point_x"],
                    params["principal_point_y"],
                    params["radial_distortion_k1"],
                ],
                camera_id=camera.camera_id,
            )
        )
    # Every keypoint of every image is a node; node offsets by database image.
    offsets = {}
    node_count = 0
    rotations = {}
    centres = {}
    for im in db_images:
        i = truth_index[im.name]
        w, x, y, z = quats[i]
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d([x, y, z, w]), np.asarray(trans[i])
        )
        reconstruction.add_image_with_trivial_frame(
            pycolmap.Image(
                name=im.name,
                keypoints=keypoints[im.image_id].astype(np.float64),
                camera_id=im.camera_id,
                image_id=im.image_id,
            ),
            cam_from_world,
        )
        rotations[im.image_id] = cam_from_world.rotation.matrix()
        centres[im.image_id] = -rotations[im.image_id].T @ np.asarray(trans[i])
        offsets[im.image_id] = node_count
        node_count += len(keypoints[im.image_id])

    # Link the verified matches into tracks.
    edges = []
    for pair_id, geometry in zip(pair_ids, geometries):
        inliers = np.asarray(geometry.inlier_matches, dtype=np.int64).reshape(-1, 2)
        a, b = pycolmap.pair_id_to_image_pair(pair_id)
        edges.append(np.stack([inliers[:, 0] + offsets[a], inliers[:, 1] + offsets[b]]))
    edges = np.concatenate(edges, axis=1)
    # Each node takes the smallest label among its neighbours' until none
    # changes, so the nodes of a connected track share one label.
    component = np.arange(node_count)
    while True:
        smaller = np.minimum(component[edges[0]], component[edges[1]])
        previous = component.copy()
        np.minimum.at(component, edges[0], smaller)
        np.minimum.at(component, edges[1], smaller)
        component = component[component]
        if np.array_equal(component, previous):
            break
    matched = np.unique(edges)
    node_image = np.empty(node_count, dtype=np.int64)
    for image_id, start in offsets.items():
        node_image[start : start + len(keypoints[image_id])] = image_id
    order = np.lexsort((matched, component[matched]))
    nodes = matched[order]
    track_of_node = component[nodes]
    starts = np.flatnonzero(np.r_[True, track_of_node[1:] != track_of_node[:-1]])
    for track_nodes in np.split(nodes, starts[1:]):
        images = node_image[track_nodes]
        if len(np.unique(images)) != len(images):
            continue
        # The point closest to all of the observations' rays.
        lhs = np.zeros((3, 3))
        rhs = np.zeros(3)
        elements = []
        for node, image_id in zip(track_nodes, images):
            feature = int(node - offsets[image_id])
            camera = reconstruction.cameras[reconstruction.images[image_id].camera_id]
            xy = camera.cam_from_img(keypoints[image_id][feature].astype(np.float64))
            ray = rotations[image_id].T @ np.array([xy[0], xy[1], 1.0])
            ray /= np.linalg.norm(ray)
            projector = np.eye(3) - np.outer(ray, ray)
            lhs += projector
            rhs += projector @ centres[image_id]
            elements.append(pycolmap.TrackElement(int(image_id), feature))
        reconstruction.add_point3D(np.linalg.solve(lhs, rhs), pycolmap.Track(elements))

    _map_at_true_poses.calls += 1
    model_dir = Path(output_path) / "0"
    model_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(model_dir))
    return {0: reconstruction}


def test_solve_reads_the_feature_folder_the_matches_file_was_matched_on(
    workspace_copy, monkeypatch
):
    workspace, matches_path, old_folder, new_folder = workspace_copy
    output = workspace / "solved.sfmr"
    # Only the mapper is replaced; matching the database, converting the model
    # and writing the .sfmr all run for real.
    _map_at_true_poses.calls = 0
    monkeypatch.setattr(pycolmap, "global_mapping", _map_at_true_poses)
    result = _invoke("solve", "-g", "--output", output, matches_path)
    assert result.exit_code == 0, result.output
    assert _map_at_true_poses.calls == 1
    assert f"Features: {old_folder}, which the .matches file" in result.output

    # The reconstruction records the .matches file's feature definition.
    recon = SfmrReconstruction.load(output)
    matches = read_matches(matches_path)
    contents = recon.metadata()["workspace"]["contents"]
    matched = matches["metadata"]["workspace"]["contents"]
    assert contents["feature_prefix_dir"] == old_folder
    assert contents["feature_options"] == matched["feature_options"]
    assert (
        contents["feature_options"]
        != (load_workspace_config(workspace)["feature_options"])
    )

    # The recorded .sift hashes are those the .matches file recorded.
    recorded = {
        name: bytes(h)
        for name, h in zip(matches["image_names"], matches["sift_content_hashes"])
    }
    for name, h in zip(recon.image_names, recon.sift_content_hashes):
        assert bytes(h) == recorded[name], name

    # Every observation's keypoint is its feature in the old folder's .sift
    # file, and not the feature at the same index in the new folder's.
    keypoints = np.asarray(recon.keypoints_xy)
    image_indexes = np.asarray(recon.track_image_indexes)
    feature_indexes = np.asarray(recon.track_feature_indexes)
    differs_from_new = 0
    for i, name in enumerate(recon.image_names):
        rows = image_indexes == i
        old = _positions(workspace / old_folder / f"{name}.sift")
        np.testing.assert_array_equal(keypoints[rows], old[feature_indexes[rows]])
        new = _positions(workspace / new_folder / f"{name}.sift")
        indexes = feature_indexes[rows]
        in_new = indexes < len(new)
        differs_from_new += int(
            np.any(new[indexes[in_new]] != old[indexes[in_new]]) or not in_new.all()
        )
    assert differs_from_new == recon.image_count

    # The points were placed through the database's keypoints at the true
    # poses, and the .sfmr reprojects them against the keypoints it read from
    # the folder it records. They agree only when the database was built from
    # the same folder: on these 8 images the mean error is 0.62 px and the
    # median 0.26 px. With the database built from the current folder instead,
    # the mean is 33 px.
    errors = np.asarray(recon.rms_reprojection_errors)
    assert len(errors) > 0
    assert float(np.mean(errors)) < 1.5
    assert float(np.median(errors)) < 0.5

    # A COLMAP export, which a bundle adjustment goes through, also reads the
    # folder the reconstruction records, not the workspace's current one.
    colmap_dir = workspace / "colmap"
    save_colmap_binary(recon, colmap_dir)
    exported = read_colmap_binary(colmap_dir)
    for name, kps in zip(exported["image_names"], exported["keypoints_per_image"]):
        old = _positions(workspace / old_folder / f"{name}.sift")
        np.testing.assert_array_equal(np.asarray(kps, dtype=np.float32)[:, :2], old)

    # `sfm epipolar` and the pose refinement of `sfm merge` read the same folder.
    name = recon.image_names[0]
    assert _require_sift_path(recon, 0, name) == workspace / old_folder / (
        f"{name}.sift"
    )
    opened = []

    def recording_sift_file(path, *args, **kwargs):
        opened.append(Path(path))
        return SiftFile(path, *args, **kwargs)

    monkeypatch.setattr(sfmtool.fileio, "SiftFile", recording_sift_file)
    merge_reconstructions([recon, recon])
    assert opened
    assert {p.parent for p in opened} == {workspace / old_folder}


def test_solve_refuses_before_solving_when_the_feature_folder_is_gone(
    workspace_copy,
):
    workspace, matches_path, old_folder, _ = workspace_copy
    shutil.rmtree(workspace / old_folder)
    output = workspace / "solved.sfmr"
    result = _invoke("solve", "-g", "--output", output, matches_path)
    assert result.exit_code != 0
    assert f"Cannot use {matches_path}" in result.output
    assert f"in the feature folder {old_folder}" in result.output
    assert "8 of the 8 .sift files it needs are missing" in result.output
    assert "Run 'sfm match' on its images again" in result.output
    # Stopped before the database was written or anything solved.
    assert "Wrote" not in result.output
    assert not output.exists()


def test_solve_refuses_when_a_sift_file_differs_from_its_recorded_hash(
    workspace_copy,
):
    workspace, matches_path, old_folder, new_folder = workspace_copy
    name = "seoul_bull_sculpture_05.jpg.sift"
    shutil.copy(workspace / new_folder / name, workspace / old_folder / name)
    output = workspace / "solved.sfmr"
    result = _invoke("solve", "-i", "--output", output, matches_path)
    assert result.exit_code != 0
    assert "1 of the 8 .sift files it needs has changed" in result.output
    assert str(workspace / old_folder / name) in result.output
    assert "Wrote" not in result.output
    assert not output.exists()


def test_to_colmap_db_refuses_when_the_feature_folder_is_gone(workspace_copy):
    workspace, matches_path, old_folder, _ = workspace_copy
    shutil.rmtree(workspace / old_folder)
    db_path = workspace / "database.db"
    result = _invoke("to-colmap-db", matches_path, db_path)
    assert result.exit_code != 0
    assert f"Cannot use {matches_path}" in result.output
    assert f"in the feature folder {old_folder}" in result.output
    assert not db_path.exists()
