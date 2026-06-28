# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Select a small, well-distributed subset of a reconstruction's cameras.

Implements ``sfm xform --include-by-distribution <COUNT>``. See
``specs/cli/xform-select-by-distribution-command.md`` for the full design.

One greedy loop driven off the reconstructed point cloud:

* **Seed.** Take the point with the widest triangulation angle over all images;
  add its observers in decreasing-observation-count order, angularly thinned so
  no two kept observers' rays toward the point are within ``H`` of each other,
  capped at ``ceil(COUNT / 3)``.
* **Loop.** Repeatedly: find the cloud point whose nearest *well-covered* point
  (>= 2 selected observers spanning >= ``H``) is farthest away, among points
  that still have an unselected observer; add that point's unselected observers,
  same decreasing-count order and same ``H`` thinning, bounded by the remaining
  budget. Stop at ``COUNT`` units.

The unit of selection is the rig frame when the reconstruction carries rig frame
data (both sensors of a 360 pair kept together), otherwise the individual image.
"""

import math
from collections import defaultdict

import numpy as np

from ..analyze.images import _compute_camera_centers
from .._sfmtool import SfmrReconstruction
from .._sfmtool.spatial import KdTree3d
from ._filter_by_image_range import _filter_images

# Thinning / well-covered angle (radians): two observers of a point count as
# "different directions" once their rays toward it are this far apart, and the
# same threshold gates a point becoming well-covered.
_H_RAD = math.radians(20.0)


class SelectByDistributionFilter:
    """Keep ``count`` strategically distributed cameras (or rig frames)."""

    def __init__(self, count: int, verbose: bool = False):
        if count < 2:
            raise ValueError(
                f"--include-by-distribution COUNT must be >= 2, got {count}"
            )
        self.count = count
        self.verbose = verbose

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        images_to_keep = _select_images(recon, self.count, verbose=self.verbose)
        images_to_keep = np.array(sorted(images_to_keep), dtype=np.uint32)
        if len(images_to_keep) == recon.image_count:
            return recon
        return _filter_images(recon, images_to_keep)

    def description(self) -> str:
        return f"Select {self.count} cameras by distribution"


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in radians between two unit vectors."""
    return float(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0)))


def _max_angle(ray: np.ndarray, rays: list[np.ndarray]) -> float:
    """Largest angle between ``ray`` and any vector in ``rays`` (0 if empty)."""
    if not rays:
        return 0.0
    return max(_angle(ray, q) for q in rays)


def _build_units(recon: SfmrReconstruction):
    """Return ``(unit_images, n_units, image_sensor, unit_label)``.

    ``unit_images[u]`` lists the image indices belonging to unit ``u``;
    ``image_sensor[i]`` is image ``i``'s sensor index within its rig (0 in the
    non-rig case); ``unit_label`` is "rig frame" or "image" for messages.
    """
    rfd = recon.rig_frame_data
    if rfd is not None:
        image_frame = np.asarray(rfd["image_frame_indexes"]).astype(np.int64)
        image_sensor = np.asarray(rfd["image_sensor_indexes"]).astype(np.int64)
        n_units = int(rfd["frames_metadata"]["frame_count"])
        unit_images: list[list[int]] = [[] for _ in range(n_units)]
        for img_i, frame_i in enumerate(image_frame):
            unit_images[int(frame_i)].append(img_i)
        return unit_images, n_units, image_sensor, "rig frame"
    n_images = recon.image_count
    return (
        [[i] for i in range(n_images)],
        n_images,
        np.zeros(n_images, dtype=np.int64),
        "image",
    )


def _triangulation_angles(
    point_ids: list[int],
    units_of_pt: dict[int, list[int]],
    ray_of: dict[int, dict[int, np.ndarray]],
) -> dict[int, float]:
    """Max pairwise ray angle over all observers, for each point."""
    out: dict[int, float] = {}
    for p in point_ids:
        rays = [ray_of[u][p] for u in units_of_pt[p]]
        best = 0.0
        for a in range(len(rays)):
            for b in range(a + 1, len(rays)):
                ang = _angle(rays[a], rays[b])
                if ang > best:
                    best = ang
        out[p] = best
    return out


def _select_images(
    recon: SfmrReconstruction, count: int, verbose: bool = False
) -> list[int]:
    track_img = np.asarray(recon.track_image_indexes).astype(np.int64)
    track_pt = np.asarray(recon.track_point_indexes).astype(np.int64)

    # Distribution is reasoned about from the finite point cloud: a point at
    # infinity is a direction, not a location, so it carries no triangulation
    # angle or position the loop can use. Drop its observations up front.
    at_infinity = np.asarray(recon.point_is_at_infinity, dtype=bool)
    finite_obs = ~at_infinity[track_pt]
    track_img = track_img[finite_obs]
    track_pt = track_pt[finite_obs]
    if len(track_pt) == 0:
        raise ValueError(
            "--include-by-distribution needs finite 3D points; this "
            "reconstruction has only points at infinity"
        )

    unit_images, n_units, image_sensor, unit_label = _build_units(recon)

    if count >= n_units:
        print(
            f"  --include-by-distribution {count}: reconstruction has only "
            f"{n_units} selectable {unit_label}s; keeping all (no-op)"
        )
        return list(range(recon.image_count))

    unit_of_image = np.empty(recon.image_count, dtype=np.int64)
    for u, imgs in enumerate(unit_images):
        for i in imgs:
            unit_of_image[i] = u

    # Per-observation viewing rays toward the observed point.
    centers = _compute_camera_centers(
        np.asarray(recon.quaternions_wxyz), np.asarray(recon.translations)
    )
    positions = np.ascontiguousarray(np.asarray(recon.positions), dtype=np.float64)
    vec = positions[track_pt] - centers[track_img]
    norms = np.linalg.norm(vec, axis=1)
    norms[norms == 0.0] = 1.0
    obs_dir = vec / norms[:, None]

    # Collapse observations to one (unit, point) ray, preferring the lowest sensor
    # index when a rig frame has more than one member image observing the point.
    unit_point_ray: dict[int, dict[int, tuple[int, np.ndarray]]] = defaultdict(dict)
    for obs_i in range(len(track_img)):
        u = int(unit_of_image[track_img[obs_i]])
        p = int(track_pt[obs_i])
        s = int(image_sensor[track_img[obs_i]])
        cur = unit_point_ray[u].get(p)
        if cur is None or s < cur[0]:
            unit_point_ray[u][p] = (s, obs_dir[obs_i])

    pts_of_unit: dict[int, list[int]] = {}
    ray_of: dict[int, dict[int, np.ndarray]] = {}
    units_of_pt: dict[int, list[int]] = defaultdict(list)
    for u, pmap in unit_point_ray.items():
        pts_of_unit[u] = list(pmap.keys())
        ray_of[u] = {p: r for p, (_s, r) in pmap.items()}
    for u in sorted(pts_of_unit):
        for p in pts_of_unit[u]:
            units_of_pt[p].append(u)

    obs_count = np.array(
        [len(pts_of_unit.get(u, ())) for u in range(n_units)], dtype=np.int64
    )
    point_ids = sorted(units_of_pt)
    tri_angle = _triangulation_angles(point_ids, units_of_pt, ray_of)
    n_pts_geh = sum(1 for p in point_ids if tri_angle[p] >= _H_RAD)
    # The bbox describes the finite point cloud; an infinity point's stored
    # position is a unit-length direction near the origin, not a location, so
    # it would skew the diagonal. Measure over finite points only.
    finite_positions = positions[~at_infinity]
    bbox = finite_positions.max(axis=0) - finite_positions.min(axis=0)
    diag = float(np.linalg.norm(bbox)) or 1.0

    _fmt = "  {:>5} | {:>4} | {:>8} | {:>9} | {:>10} | {:>9} | {:>9} | {:>7} | {:>6}"

    def vrow(step: str, target, dist, ang, added, targetable):
        if not verbose:
            return
        n_unsel = sum(1 for c in unsel_obs.values() if c > 0)
        print(
            _fmt.format(
                step,
                len(selected),
                len(well_covered),
                n_unsel,
                "--" if targetable is None else len(targetable),
                "--" if target is None else target,
                "--" if dist is None else f"{dist / diag:.3f}",
                "--" if ang is None else f"{math.degrees(ang):.1f}",
                added,
            )
        )

    if verbose:
        print(
            f"  verbose trace (H={math.degrees(_H_RAD):.0f} deg; {len(point_ids)} "
            f"points, {n_pts_geh} with full triangulation angle >= H; "
            f"bbox diag {diag:.3g})"
        )
        print(
            "    |S| = selected units; well-cov = points with >=2 selected observers "
            ">= H apart;\n"
            "    has-unsel = points still having an unselected observer; "
            "targetable = those with full tri-angle >= H;\n"
            "    target_pt / dist/diag / tgt_ang = the farthest-point pick this step"
        )
        print(
            _fmt.format(
                "step",
                "|S|",
                "well-cov",
                "has-unsel",
                "targetable",
                "target_pt",
                "dist/diag",
                "tgt_ang",
                "+units",
            )
        )

    def observer_order(unit_list: list[int]) -> list[int]:
        # Decreasing observation count, ties by lowest unit index.
        return sorted(unit_list, key=lambda v: (-int(obs_count[v]), v))

    selected: list[int] = []
    selected_set: set[int] = set()
    sel_rays: dict[int, list[np.ndarray]] = {p: [] for p in point_ids}
    sel_par: dict[int, float] = defaultdict(float)
    well_covered: set[int] = set()
    covered_positions: list[np.ndarray] = []
    unsel_obs: dict[int, int] = {p: len(units_of_pt[p]) for p in point_ids}

    def add_unit(u: int) -> None:
        selected.append(u)
        selected_set.add(u)
        for p in pts_of_unit.get(u, ()):
            unsel_obs[p] -= 1
            ru = ray_of[u][p]
            rays = sel_rays[p]
            new_par = max(sel_par[p], _max_angle(ru, rays))
            rays.append(ru)
            sel_par[p] = new_par
            if p not in well_covered and len(rays) >= 2 and new_par >= _H_RAD:
                well_covered.add(p)
                covered_positions.append(positions[p])

    def thinned_keep(target_p: int, candidates: list[int], budget: int) -> list[int]:
        """Keep candidates whose ray toward ``target_p`` opens a new angle.

        A candidate is kept if its ray is >= H from every already-selected
        observer's ray on the point and from every candidate kept so far. If
        nothing qualifies, keep the single candidate that adds the most parallax
        on the point (drives it toward well-covered fastest) so the loop makes
        progress; ``candidates`` is already in (most observations, lowest index)
        order, so ties there resolve deterministically.
        """
        if budget <= 0 or not candidates:
            return []
        existing = sel_rays[target_p]
        kept: list[int] = []
        for v in candidates:
            if len(kept) >= budget:
                break
            rv = ray_of[v][target_p]
            if existing and _max_angle(rv, existing) < _H_RAD:
                continue
            if any(_angle(rv, ray_of[k][target_p]) < _H_RAD for k in kept):
                continue
            kept.append(v)
        if not kept:
            kept = [
                max(candidates, key=lambda v: _max_angle(ray_of[v][target_p], existing))
            ]
        return kept

    # ---- Seed: widest triangulation angle ------------------------------------
    seed_p = max(point_ids, key=lambda p: (tri_angle[p], -p))
    seed_cap = min(max(1, math.ceil(count / 3)), count)
    seed_units = thinned_keep(seed_p, observer_order(units_of_pt[seed_p]), seed_cap)
    for u in seed_units:
        add_unit(u)
    vrow("seed", seed_p, None, tri_angle[seed_p], len(seed_units), None)
    if not verbose:
        print(
            f"  Seed: {len(selected)} {unit_label}(s) around the widest-baseline "
            f"point (triangulation angle {math.degrees(tri_angle[seed_p]):.1f} deg)"
        )

    # ---- Main loop: farthest-point coverage ----------------------------------
    # A point is a candidate target only if it can still gain a new observer AND
    # its observers span enough baseline to ever be well-covered (tri_angle >= H);
    # otherwise visiting it just burns budget on an un-triangulatable point.
    targetable = [p for p in point_ids if tri_angle[p] >= _H_RAD]
    fps_steps = 0
    while len(selected) < count:
        improvable = [p for p in targetable if unsel_obs[p] > 0]
        if not improvable:
            break

        if covered_positions:
            tree = KdTree3d(np.ascontiguousarray(covered_positions, dtype=np.float64))
            query = np.ascontiguousarray(
                [positions[p] for p in improvable], dtype=np.float64
            )
            nn_idx = np.asarray(tree.nearest(query))
            nn_pos = np.asarray(covered_positions)[nn_idx]
            dists = np.linalg.norm(query - nn_pos, axis=1)
            best_i = max(
                range(len(improvable)), key=lambda i: (float(dists[i]), -improvable[i])
            )
            target_p = improvable[best_i]
            target_d = float(dists[best_i])
        else:
            target_p = max(improvable, key=lambda p: (tri_angle[p], -p))
            target_d = None

        candidates = observer_order(
            [v for v in units_of_pt[target_p] if v not in selected_set]
        )
        kept = thinned_keep(target_p, candidates, count - len(selected))
        for u in kept:
            add_unit(u)
        fps_steps += 1
        vrow(
            str(fps_steps),
            target_p,
            target_d,
            tri_angle[target_p],
            len(kept),
            improvable,
        )

    # ---- Fill any leftover budget (every point already covered) --------------
    def not_covered_count(v: int) -> int:
        return sum(1 for p in pts_of_unit.get(v, ()) if p not in well_covered)

    n_fill = 0
    while len(selected) < count:
        u = max(
            (v for v in range(n_units) if v not in selected_set),
            key=lambda v: (not_covered_count(v), int(obs_count[v]), -v),
        )
        add_unit(u)
        n_fill += 1
    if n_fill:
        vrow("fill", None, None, None, n_fill, None)

    print(
        f"  Coverage loop: {fps_steps} farthest-point step(s); "
        f"{len(well_covered)} of {len(point_ids)} points well-covered "
        f"(>= 2 observers, >= {math.degrees(_H_RAD):.0f} deg)"
    )
    return _images_for_units(selected, unit_images)


def _images_for_units(units: list[int], unit_images: list[list[int]]) -> list[int]:
    keep: list[int] = []
    for u in units:
        keep.extend(unit_images[u])
    return keep
