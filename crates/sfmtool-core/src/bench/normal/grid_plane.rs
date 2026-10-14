// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! [`grid_plane_normal`]: the patch's normal from the plane through the nine
//! cells of the ZNCC grid, each registered by a shift search on one render
//! per view, gated in each view, and triangulated from the views it survived
//! in.
//!
//! Each `in` view is rendered once at the track's current frame as a context
//! tile: the patch re-anchored in its plane on the view's keypoint and drawn
//! through that plane ([`WarpMap::from_patch`]) on the working grid plus a
//! margin of the cell search bound, the render the keypoint localizer makes
//! for its context tiles. The reference view's render, without the margin,
//! gives the nine templates. The cell registration kernel
//! (`crate::patch::cell_registration`) registers and gates every cell of
//! every other view and fits each view's robust affine map; the cell plane
//! kernel (`crate::patch::cell_plane_fit`) triangulates the cells that
//! survive and fits the plane with its determinacy verdict. The patch is
//! turned toward that plane with [`tilt_patch`] and measured again, under the
//! acceptance and stopping rules of the piecewise refinement's loop.
//!
//! `specs/core/bench/editable-track.md` § "Estimating the normal" describes
//! the step; `specs/drafts/piece-gated-grid-normal.md` is the design it is
//! built from.

use nalgebra::Vector3;

use crate::camera::sampler::{render_phase, render_tile};
use crate::camera::warp_map::{patch_grid_jacobian, singular_values_2x2};
use crate::camera::WarpMap;
use crate::patch::cell_plane_fit::{fit_cell_plane, CellRay, PosedView};
use crate::patch::cell_plane_normals::{
    CellPlaneNormal, CellPlaneParams, CellPlaneStatus, NormalDeterminacy,
};
use crate::patch::cell_registration::{
    fit_pass, read_cells, CellGateOptions, CellLayout, CellReading, CellTemplate, CellTemplates,
    WorkingPatch,
};
use crate::patch::cloud::OrientedPatch;
use crate::patch::cluster_refine::{CellStatus, LoopStop, PiecewiseParams, ACCEPT_ZNCC_TOLERANCE};
use crate::patch::keypoint_localize::{resolve_reference, shifted_center};
use crate::patch::normal_refine::ProjectedImage;
use crate::patch::reference_view::ReferenceViewOptions;
use crate::progress::Progress;
use crate::progress_note;
use crate::reconstruction::edited::EditedReconstruction;

use super::super::evaluate::{check_views, seed_of, stored_patch_resolution, track_reference};
use super::super::fit::FitOptions;
use super::super::steps::tilt_patch;
use super::super::track::EditableTrack;
use super::{
    frame_of, normal_preconditions, perpendicular_part, turn_and_read, NormalError, NormalEstimate,
    NormalReport,
};

/// The smallest working grid the step renders: three samples a side are the
/// fewest that cut into nine cells.
const MIN_WORKING_RESOLUTION: u32 = 3;

/// A sample variance under this is flat for the acceptance score.
const FLAT_VARIANCE: f64 = 1e-9;

/// How [`grid_plane_normal`] measures the cells and when its loop stops.
#[derive(Debug, Clone)]
pub struct GridPlaneNormalOptions {
    /// Search bound for a cell's shift, grid px of the working grid; also the
    /// margin, rounded up, each view's context tile is rendered with. The
    /// piecewise refinement's
    /// [`cell_shift_bound_px`](PiecewiseParams::cell_shift_bound_px), 2.0.
    pub cell_shift_bound_px: f32,
    /// A cell whose ZNCC at its optimum is below this is refused in that view.
    /// [`DEFAULT_MIN_CELL_ZNCC`](crate::patch::cluster_refine::DEFAULT_MIN_CELL_ZNCC),
    /// 0.8.
    pub min_cell_zncc: f32,
    /// A cell whose ZNCC peak is flatter than this along its flattest
    /// direction is refused in that view, in ZNCC per grid px².
    /// [`DEFAULT_MIN_CELL_CURVATURE`](crate::patch::cluster_refine::DEFAULT_MIN_CELL_CURVATURE),
    /// 0.02.
    pub min_cell_curvature: f32,
    /// Side of the working grid the views are rendered on, in samples, at
    /// least 3. `None` renders at the reconstruction's own patch resolution
    /// where it stores patch bitmaps, and otherwise at the fit's sub-pixel
    /// resolution, the grid the track's bitmap is rendered on.
    pub working_resolution: Option<u32>,
    /// Triangulation, floors, Tukey loop and determinacy, as the cell plane
    /// normal kernel reads them.
    pub plane: CellPlaneParams,
    /// The loop stops when a turn moves no cell centre by more than this, in
    /// grid px.
    pub update_tolerance_px: f32,
    /// The most renders the loop makes, at least 1. A turn is kept only once
    /// a render at the turned frame has scored it, so the last render's own
    /// turn is not taken unless it is under
    /// [`Self::update_tolerance_px`].
    pub max_iterations: u8,
    /// The reading the step ends with, the sampler the context tiles are
    /// rendered with (`refine.sampler`), and the reference-view rule's
    /// grazing gate and sampler (`localize`).
    pub fit: FitOptions,
}

impl Default for GridPlaneNormalOptions {
    fn default() -> Self {
        let piecewise = PiecewiseParams::default();
        Self {
            cell_shift_bound_px: piecewise.cell_shift_bound_px,
            min_cell_zncc: piecewise.min_cell_zncc,
            min_cell_curvature: piecewise.min_cell_curvature,
            working_resolution: None,
            plane: CellPlaneParams::default(),
            update_tolerance_px: piecewise.update_tolerance_px,
            max_iterations: piecewise.max_iterations,
            fit: FitOptions::default(),
        }
    }
}

/// What the cells said, carried by [`NormalEstimate::GridPlaneCells`]. Every
/// reading is of the last render the loop kept: the render at the frame the
/// step turns the patch to, or, when the loop converged, the render one turn
/// of under [`GridPlaneNormalOptions::update_tolerance_px`] before it.
#[derive(Debug, Clone, PartialEq)]
pub struct GridPlaneCells {
    /// Which axes of the normal the cells fixed.
    pub determinacy: NormalDeterminacy,
    /// What became of each cell in the triangulation and the plane fit,
    /// `[row * 3 + col]` from the top-left cell.
    pub cell_status: [CellPlaneStatus; 9],
    /// The rays each cell was triangulated from, the reference's among them.
    pub cell_rays: [u32; 9],
    /// Each cell's weight in the plane fit; `0` where it has none.
    pub cell_weight: [f64; 9],
    /// Per observation of the track, each cell's registration status in that
    /// view. The reference's row reads [`CellStatus::Fitted`] for a cell its
    /// template can be registered with, [`CellStatus::RefusedCurvature`] for
    /// a flat one and [`CellStatus::NotAttempted`] for one partly off its
    /// photograph; a row that is not `in`, repeats an image, or could not be
    /// rendered reads [`CellStatus::NotAttempted`] throughout.
    pub view_cells: Vec<[CellStatus; 9]>,
    /// The weighted rms distance of the cells from their plane, world units;
    /// `NaN` when no plane was fitted.
    pub plane_rms: f64,
    /// Signed distance of the fitted plane from the patch centre along the
    /// fitted normal, world units; `NaN` when the cells fixed no axis. The
    /// step does not move the centre.
    pub plane_offset: f64,
    /// The renders the loop made.
    pub iterations: u8,
    /// Why the loop stopped. [`LoopStop::NotRun`] when no reference could be
    /// rendered or no view could be scored against it, and
    /// [`LoopStop::Measured`] when the last render's cells fixed no axis, so
    /// no turn was proposed.
    pub stop: LoopStop,
    /// The acceptance score at the frame the step started from: the mean over
    /// the non-reference views of the ZNCC of each view's whole tile against
    /// the reference's render. `NaN` when no view scored.
    pub score_before: f64,
    /// The acceptance score at the last render the loop kept, the render the
    /// other readings are of.
    pub score_after: f64,
}

impl GridPlaneCells {
    /// How many cells the plane fit gave weight.
    pub fn in_plane(&self) -> usize {
        self.cell_status
            .iter()
            .filter(|&&s| s == CellPlaneStatus::InPlane)
            .count()
    }
}

impl std::fmt::Display for GridPlaneCells {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} of 9 cells in the plane", self.in_plane())?;
        match self.determinacy {
            NormalDeterminacy::BothAxes => write!(
                f,
                ", both axes fixed, rms {:.4} off the plane",
                self.plane_rms
            )?,
            NormalDeterminacy::OneAxis { .. } => write!(f, ", one axis fixed")?,
            NormalDeterminacy::None => write!(f, ", no axis fixed")?,
        }
        let stop = match self.stop {
            LoopStop::NotRun => "not run",
            LoopStop::Converged => "converged",
            LoopStop::Cap => "at the render cap",
            LoopStop::Rejected => "last turn rejected",
            LoopStop::Oscillation => "oscillating",
            LoopStop::Measured => "nothing to turn to",
        };
        let renders = if self.iterations == 1 {
            "render"
        } else {
            "renders"
        };
        write!(f, ", {} {renders}, {stop}", self.iterations)
    }
}

/// Turn the track's patch to the plane through its cells.
///
/// The patch's square is cut into the nine cells of the ZNCC grid. Each `in`
/// view is rendered once at the current frame, centred on its keypoint and
/// drawn through the patch's plane, and each cell of it is registered against
/// the same cell of the reference view's render by a shift search within
/// [`GridPlaneNormalOptions::cell_shift_bound_px`]. A cell is refused in a view
/// for curvature, for ZNCC, for the bound, or as an outlier of the view's
/// robust affine fit, by the piecewise refinement's rules. Every surviving
/// cell is a ray through the pixel its registered position maps to, the
/// reference contributes a ray through each cell's centre, and the cells are
/// triangulated and fitted with a weighted, robust plane that says whether it
/// fixes both axes of the normal, one, or none.
///
/// Under both axes the patch is turned to the plane's normal, signed to keep
/// the face the patch shows; under one, only about the axis the cells fix,
/// keeping its rotation about the line they lie along; under none, not at
/// all. Each turn is rendered and measured again, and kept only when the mean
/// whole-tile ZNCC of the views against the reference does not fall by more
/// than [`ACCEPT_ZNCC_TOLERANCE`], nor below its starting value. The loop
/// stops when a turn moves no cell centre by more than
/// [`GridPlaneNormalOptions::update_tolerance_px`] (that turn is taken), at
/// [`GridPlaneNormalOptions::max_iterations`] renders, when a turn moves the
/// cells no less than the one before (the better-scoring of the two frames
/// is kept), or when a turn is rejected.
///
/// The step ends as the other normal steps do: the original track is turned
/// to the kept frame's normal with [`tilt_patch`] and read back, its bitmap
/// rendered. The centre does not move.
///
/// # Example
///
/// ```no_run
/// # use sfmtool_core::bench::{grid_plane_normal, GridPlaneNormalOptions};
/// # use sfmtool_core::progress::Progress;
/// # fn run(
/// #     track: &sfmtool_core::bench::EditableTrack,
/// #     edited: &sfmtool_core::EditedReconstruction,
/// #     images: &[sfmtool_core::patch::normal_refine::ProjectedImage<'_>],
/// # ) -> Result<(), Box<dyn std::error::Error>> {
/// let (turned, report) = grid_plane_normal(
///     track,
///     edited,
///     images,
///     &GridPlaneNormalOptions::default(),
///     &Progress::none(),
/// )?;
/// println!("{report}"); // "9 of 9 cells in the plane, both axes fixed, …"
/// # let _ = turned;
/// # Ok(())
/// # }
/// ```
pub fn grid_plane_normal(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &GridPlaneNormalOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, NormalReport), NormalError> {
    check_views(edited, images)?;
    normal_preconditions(track)?;
    let resolution = options
        .working_resolution
        .or_else(|| stored_patch_resolution(&edited.base))
        .unwrap_or(options.fit.refine.resolution)
        .max(MIN_WORKING_RESOLUTION);
    let cells = Cells::new(track, images, resolution, options, progress);

    let end = {
        let mut phase = progress.phase("grid plane normal");
        let tolerance = f64::from(options.update_tolerance_px);
        let end = turn_loop(
            track.clone(),
            options.max_iterations,
            tolerance,
            |t: &EditableTrack| cells.measure(t),
            |m: &Measurement| m.score,
            |t: &EditableTrack, m: &Measurement| cells.propose(t, edited, m),
        )?;
        progress_note!(phase, "{} renders", end.iterations);
        end
    };

    let kept = frame_of(&end.state);
    let measured = &end.measured;
    let report = GridPlaneCells {
        determinacy: measured.plane.determinacy,
        cell_status: measured.plane.cell_status,
        cell_rays: measured.plane.cell_rays,
        cell_weight: measured.plane.cell_weight,
        view_cells: measured.view_cells.clone(),
        plane_rms: measured.plane.plane_rms,
        plane_offset: plane_offset(&measured.plane, &kept),
        iterations: end.iterations,
        stop: end.stop,
        score_before: end.start_score,
        score_after: measured.score,
    };
    turn_and_read(
        track,
        edited,
        images,
        kept.normal(),
        NormalEstimate::GridPlaneCells(report),
        &options.fit,
        progress,
    )
}

/// What every render of one [`grid_plane_normal`] shares.
struct Cells<'a, 'v> {
    images: &'a [ProjectedImage<'v>],
    options: &'a GridPlaneNormalOptions,
    progress: &'a Progress<'a>,
    layout: CellLayout,
    gates: CellGateOptions,
    /// The `in` rows the step renders, one per image, in row order.
    rows: Vec<usize>,
    /// The reference row, from `rows`; `None` when the rule found none.
    reference: Option<usize>,
    /// The farthest a cell centre lies from the grid's middle, grid px.
    reach: f64,
}

/// One render of every view at one frame, and what the cells said.
struct Measurement {
    plane: CellPlaneNormal,
    view_cells: Vec<[CellStatus; 9]>,
    /// The acceptance score, `NaN` when no view scored.
    score: f64,
}

/// One view's context tile.
struct Render {
    /// The patch re-anchored on the view's keypoint, at the working grid's
    /// size: what the cell positions are read on.
    placement: OrientedPatch,
    /// `side × side × channels`, interleaved, `NaN` off the photograph.
    samples: Vec<f32>,
    side: usize,
    channels: usize,
}

impl<'a, 'v> Cells<'a, 'v> {
    fn new(
        track: &EditableTrack,
        images: &'a [ProjectedImage<'v>],
        resolution: u32,
        options: &'a GridPlaneNormalOptions,
        progress: &'a Progress<'a>,
    ) -> Self {
        let layout = CellLayout::new(resolution, options.cell_shift_bound_px);
        let gates = CellGateOptions {
            min_cell_zncc: options.min_cell_zncc,
            min_cell_curvature: options.min_cell_curvature,
        };
        // One row per image, as the photometric step reads them: a second
        // sighting in one image adds no view.
        let mut seen = std::collections::BTreeSet::new();
        let rows: Vec<usize> = track
            .in_observations()
            .into_iter()
            .filter(|&i| seen.insert(track.observations[i].image))
            .collect();
        let view_set: Vec<u32> = rows.iter().map(|&i| track.observations[i].image).collect();
        let seeds: Vec<Option<[f64; 2]>> = rows
            .iter()
            .map(|&i| seed_of(&track.observations[i]))
            .collect();
        // The fit's reference: the one the track holds where its row is `in`
        // with a keypoint, and otherwise the reference-view rule's pick.
        let reference = resolve_reference(
            &frame_of(track),
            images,
            &view_set,
            &seeds,
            track_reference(track, &rows),
            options.fit.localize.min_grazing_cos,
            resolution,
            options.fit.localize.sampler,
            &ReferenceViewOptions::default(),
            progress,
        )
        .map(|k| rows[k]);
        let reach = layout
            .centres
            .iter()
            .flatten()
            .map(|c| c[0].hypot(c[1]))
            .fold(0.0, f64::max);
        Self {
            images,
            options,
            progress,
            layout,
            gates,
            rows,
            reference,
            reach,
        }
    }

    /// Render every view at `track`'s frame, register and gate its cells,
    /// triangulate them and fit the plane, and score the views against the
    /// reference.
    fn measure(&self, track: &EditableTrack) -> Result<Measurement, NormalError> {
        let frame = frame_of(track);
        let r = self.layout.resolution;
        let wpp = 2.0 * frame.half_extent[0] / r as f64;
        let params = &self.options.plane;
        let mut view_cells = vec![[CellStatus::NotAttempted; 9]; track.observations.len()];
        let mut rays: [Vec<CellRay>; 9] = Default::default();
        let nothing = |view_cells| Measurement {
            plane: fit_cell_plane(&Default::default(), params),
            view_cells,
            score: f64::NAN,
        };

        let Some(reference) = self.reference else {
            return Ok(nothing(view_cells));
        };
        self.progress.check_cancel()?;
        let ref_observation = &track.observations[reference];
        let ref_view = &self.images[ref_observation.image as usize];
        let Some(ref_render) = self.render(&frame, ref_view, seed_of(ref_observation), 0) else {
            return Ok(nothing(view_cells));
        };
        let planes = planar(&ref_render.samples, r * r, ref_render.channels);
        let templates = CellTemplates::from_planes(&planes, ref_render.channels, &self.layout);
        let src_channels: Vec<usize> = (0..ref_render.channels).collect();
        let ref_posed = PosedView::new(ref_view.camera, ref_view.cam_from_world);
        for (k, centre) in self.layout.centres.iter().flatten().enumerate() {
            let status = template_status(&templates.cells[k / 3][k % 3]);
            view_cells[reference][k] = status;
            if status == CellStatus::Fitted {
                rays[k].extend(cell_ray(
                    &ref_render.placement,
                    ref_view,
                    &ref_posed,
                    *centre,
                    wpp,
                    params,
                ));
            }
        }

        let mut scores: Vec<f64> = Vec::with_capacity(self.rows.len());
        for &row in &self.rows {
            if row == reference {
                continue;
            }
            // The cancel flag is polled per render: the render is the one read
            // of the photograph.
            self.progress.check_cancel()?;
            let observation = &track.observations[row];
            let view = &self.images[observation.image as usize];
            let Some(render) = self.render(&frame, view, seed_of(observation), self.layout.margin)
            else {
                continue;
            };
            if let Some(z) = tile_zncc(&planes, ref_render.channels, &render, self.layout.margin) {
                scores.push(z);
            }
            let patch = WorkingPatch::new(render.side, render.channels, render.samples.clone());
            let readings = read_cells(&templates, &src_channels, &patch, &self.layout, &self.gates);
            let statuses = gated(readings, &self.layout);
            let posed = PosedView::new(view.camera, view.cam_from_world);
            for (k, (status, reading)) in statuses
                .iter()
                .flatten()
                .zip(readings.iter().flatten())
                .enumerate()
            {
                view_cells[row][k] = *status;
                let usable = *status == CellStatus::Fitted
                    || (params.include_refused_outlier && *status == CellStatus::RefusedOutlier);
                if !usable {
                    continue;
                }
                let c = self.layout.centres[k / 3][k % 3];
                let at = [c[0] + reading.shift[0], c[1] + reading.shift[1]];
                rays[k].extend(cell_ray(&render.placement, view, &posed, at, wpp, params));
            }
        }
        let score = if scores.is_empty() {
            f64::NAN
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };
        Ok(Measurement {
            plane: fit_cell_plane(&rays, params),
            view_cells,
            score,
        })
    }

    /// The turn the cells of `measured` ask of `track`, and the farthest it
    /// moves a cell centre, in grid px; `None` when they fix no axis.
    fn propose(
        &self,
        track: &EditableTrack,
        edited: &EditedReconstruction,
        measured: &Measurement,
    ) -> Result<Option<(EditableTrack, f64)>, NormalError> {
        let was = frame_of(track).normal();
        let target = match measured.plane.determinacy {
            NormalDeterminacy::None => return Ok(None),
            NormalDeterminacy::BothAxes => {
                let n = Vector3::from(measured.plane.normal);
                // A plane has no sign: keep the face the patch shows.
                if n.dot(&was) < 0.0 {
                    -n
                } else {
                    n
                }
            }
            NormalDeterminacy::OneAxis { free_axis } => {
                let line = Vector3::from(free_axis);
                let length = line.norm();
                if length.is_nan() || length <= 0.0 {
                    return Ok(None);
                }
                perpendicular_part(was, line / length).unwrap_or(was)
            }
        };
        if !target.iter().all(|c| c.is_finite()) {
            return Ok(None);
        }
        let (turned, tilt) = tilt_patch(track, edited, target)?;
        // A turn by `θ` about the centre carries a cell centre `ρ` grid px
        // from it through a chord of `2ρ sin(θ/2)`.
        let largest_move = 2.0 * self.reach * (tilt.degrees.to_radians() / 2.0).sin();
        Ok(Some((turned, largest_move)))
    }

    /// One view's context tile at `frame`: the patch re-anchored on
    /// `keypoint` (on the frame itself where there is none or its ray misses
    /// the plane), widened by `margin` grid px on every side, and drawn
    /// through its plane with the sampler the fit's bitmap render picks from
    /// the view's Jacobian. `None` when no sample lands on the photograph.
    fn render(
        &self,
        frame: &OrientedPatch,
        view: &ProjectedImage<'_>,
        keypoint: Option<[f64; 2]>,
        margin: usize,
    ) -> Option<Render> {
        let r = self.layout.resolution;
        let placement = keypoint
            .and_then(|kp| frame.anchored_at_keypoint(view.camera, view.cam_from_world, kp))
            .unwrap_or_else(|| frame.clone());
        let jacobian = patch_grid_jacobian(&placement, view.camera, view.cam_from_world, r);
        let sampler = self.options.fit.refine.sampler.for_jacobian(jacobian);
        let side = r + 2 * margin;
        let scale = side as f64 / r as f64;
        let mut context = placement.clone();
        context.half_extent = [
            placement.half_extent[0] * scale,
            placement.half_extent[1] * scale,
        ];
        let mut map = WarpMap::from_patch(&context, view.camera, view.cam_from_world, side as u32);
        let tile = {
            let _phase = render_phase(self.progress, sampler, 1);
            render_tile(view.pyramid, &mut map, sampler)
        };
        // A fourth channel is the photograph's alpha, which carries no
        // texture.
        let channels = (tile.channels() as usize).clamp(1, 3);
        let mut samples = vec![f32::NAN; side * side * channels];
        let mut any = false;
        for row in 0..side as u32 {
            for col in 0..side as u32 {
                if !map.is_valid(col, row) {
                    continue;
                }
                any = true;
                let at = (row as usize * side + col as usize) * channels;
                for c in 0..channels {
                    samples[at + c] = f32::from(tile.get_pixel(col, row, c as u32));
                }
            }
        }
        any.then_some(Render {
            placement,
            samples,
            side,
            channels,
        })
    }
}

/// A reference cell's status as a template: refused for curvature where it
/// is flat in every channel, not attempted where some sample of it is off the
/// photograph, and fitted otherwise.
fn template_status(template: &CellTemplate) -> CellStatus {
    if template.missing {
        CellStatus::NotAttempted
    } else if template.channels.iter().all(Option::is_none) {
        CellStatus::RefusedCurvature
    } else {
        CellStatus::Fitted
    }
}

/// The statuses of one view's readings after the view's robust affine fit: a
/// fitted cell the fit gives no weight is an outlier, and where the fit fails
/// (its map reflects or is not finite) no cell of the view is kept, as the
/// piecewise refinement stores a failed fit.
fn gated(readings: [[CellReading; 3]; 3], layout: &CellLayout) -> [[CellStatus; 3]; 3] {
    let mut statuses = readings.map(|line| line.map(|reading| reading.status));
    let pass = fit_pass(readings, layout);
    for (row, line) in statuses.iter_mut().enumerate() {
        for (col, status) in line.iter_mut().enumerate() {
            if *status != CellStatus::Fitted {
                continue;
            }
            *status = match &pass {
                Some(pass) if pass.weights[row][col] > 0.0 => CellStatus::Fitted,
                Some(_) => CellStatus::RefusedOutlier,
                None => CellStatus::NotAttempted,
            };
        }
    }
    statuses
}

/// The ray through the photograph's pixel under grid position `at` of a
/// render through `placement`, in grid px centred on the grid. Its noise floor
/// is a measured shift's precision carried into the view's pixels by the
/// render map's local Jacobian, floored at the kernel's absolute floor.
/// `None` where the position does not project.
fn cell_ray(
    placement: &OrientedPatch,
    view: &ProjectedImage<'_>,
    posed: &PosedView<'_>,
    at: [f64; 2],
    wpp: f64,
    params: &CellPlaneParams,
) -> Option<CellRay> {
    let pixel = |gx: f64, gy: f64| {
        let world = shifted_center(placement, gx, gy, wpp, wpp);
        view.camera
            .project_homogeneous(view.cam_from_world, world.coords, placement.w)
    };
    let centre = pixel(at[0], at[1])?;
    let (right, left) = (pixel(at[0] + 0.5, at[1])?, pixel(at[0] - 0.5, at[1])?);
    let (down, up) = (pixel(at[0], at[1] + 0.5)?, pixel(at[0], at[1] - 0.5)?);
    let jacobian = [
        [right[0] - left[0], down[0] - up[0]],
        [right[1] - left[1], down[1] - up[1]],
    ];
    let largest = singular_values_2x2(jacobian)[0];
    let floor = (params.cell_shift_precision_grid_px * largest).max(params.ray_noise_floor_px);
    posed.ray_through(centre, floor)
}

/// Interleaved `n × channels` samples as `channels` planes of `n`.
fn planar(samples: &[f32], n: usize, channels: usize) -> Vec<f32> {
    let mut planes = vec![f32::NAN; n * channels];
    for (k, px) in samples.chunks_exact(channels).enumerate() {
        for (c, &v) in px.iter().enumerate() {
            planes[c * n + k] = v;
        }
    }
    planes
}

/// The ZNCC of a view's tile at zero shift, its working grid's middle `R×R`,
/// against the reference's render `planes`, over the samples both carry,
/// averaged over the channels the reference is not flat in; a channel the
/// view is flat in scores `0`. `None` when no channel scores.
fn tile_zncc(planes: &[f32], ref_channels: usize, render: &Render, margin: usize) -> Option<f64> {
    let side = render.side;
    let r = side - 2 * margin;
    let n = r * r;
    let (mut sum, mut scored) = (0.0, 0usize);
    for c in 0..ref_channels.min(render.channels) {
        let plane = &planes[c * n..][..n];
        let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(n);
        for y in 0..r {
            for x in 0..r {
                let a = plane[y * r + x];
                let b = render.samples[((y + margin) * side + x + margin) * render.channels + c];
                if a.is_finite() && b.is_finite() {
                    pairs.push((f64::from(a), f64::from(b)));
                }
            }
        }
        if pairs.len() < 2 {
            continue;
        }
        let count = pairs.len() as f64;
        let (ma, mb) = pairs
            .iter()
            .fold((0.0, 0.0), |(sa, sb), (a, b)| (sa + a, sb + b));
        let (ma, mb) = (ma / count, mb / count);
        let (mut saa, mut sbb, mut sab) = (0.0, 0.0, 0.0);
        for (a, b) in &pairs {
            let (da, db) = (a - ma, b - mb);
            saa += da * da;
            sbb += db * db;
            sab += da * db;
        }
        if saa / count < FLAT_VARIANCE {
            continue;
        }
        scored += 1;
        if sbb / count >= FLAT_VARIANCE {
            sum += sab / (saa * sbb).sqrt();
        }
    }
    (scored > 0).then(|| sum / scored as f64)
}

/// The signed distance of `plane`'s weighted cell centroid from `frame`'s
/// centre along its normal; `NaN` when the cells fixed no axis.
fn plane_offset(plane: &CellPlaneNormal, frame: &OrientedPatch) -> f64 {
    if plane.determinacy == NormalDeterminacy::None {
        return f64::NAN;
    }
    let normal = Vector3::from(plane.normal);
    let (mut centroid, mut total) = (Vector3::zeros(), 0.0);
    for (position, &w) in plane.cell_positions.iter().zip(&plane.cell_weight) {
        if w > 0.0 {
            centroid += Vector3::from(*position) * w;
            total += w;
        }
    }
    if total.is_nan() || total <= 0.0 {
        return f64::NAN;
    }
    (centroid / total - frame.center.coords).dot(&normal)
}

/// How [`turn_loop`] ended.
struct LoopEnd<T, M> {
    /// The state kept.
    state: T,
    /// The last measurement of the state kept.
    measured: M,
    /// Measurements made.
    iterations: u8,
    stop: LoopStop,
    /// The score of the starting state.
    start_score: f64,
}

/// The piecewise refinement's two-level loop with the measurement and the
/// turn behind closures: `measure` renders and reads a state, `score` is its
/// acceptance score, and `propose` gives the turned state the measurement
/// asks for with the farthest the turn moves a cell centre, or `None` when it
/// asks for none.
///
/// The loop measures the start, then until it stops proposes a turn and
/// measures the turned state. A turn that moves no cell centre by more than
/// `tolerance` is taken without a render ([`LoopStop::Converged`]); at
/// `max_iterations` measurements the turn the last one asks is not taken,
/// having no render to score it ([`LoopStop::Cap`]); a turn whose score falls
/// below the current one by more than [`ACCEPT_ZNCC_TOLERANCE`], or below the
/// start's, is not taken ([`LoopStop::Rejected`]); a turn that moves the
/// cells no less than the one before ends the loop on whichever of the two
/// states scores higher ([`LoopStop::Oscillation`]). A start that does not
/// score stops at once ([`LoopStop::NotRun`]), and a measurement that asks for
/// no turn stops where it is ([`LoopStop::Measured`]).
fn turn_loop<T, M, E>(
    start: T,
    max_iterations: u8,
    tolerance: f64,
    mut measure: impl FnMut(&T) -> Result<M, E>,
    score: impl Fn(&M) -> f64,
    mut propose: impl FnMut(&T, &M) -> Result<Option<(T, f64)>, E>,
) -> Result<LoopEnd<T, M>, E> {
    let max_iterations = max_iterations.max(1);
    let mut state = start;
    let mut measured = measure(&state)?;
    let mut iterations = 1u8;
    let start_score = score(&measured);
    let end = |state, measured, iterations, stop| {
        Ok(LoopEnd {
            state,
            measured,
            iterations,
            stop,
            start_score,
        })
    };
    if !start_score.is_finite() {
        return end(state, measured, iterations, LoopStop::NotRun);
    }
    let mut current = start_score;
    let mut previous_move = f64::INFINITY;
    loop {
        let Some((next, largest_move)) = propose(&state, &measured)? else {
            return end(state, measured, iterations, LoopStop::Measured);
        };
        if largest_move <= tolerance {
            return end(next, measured, iterations, LoopStop::Converged);
        }
        if iterations >= max_iterations {
            return end(state, measured, iterations, LoopStop::Cap);
        }
        let next_measured = measure(&next)?;
        iterations += 1;
        let next_score = score(&next_measured);
        // The floor: the current score less the tolerance, but never below
        // the start's, so the tolerance cannot accumulate over several turns.
        let floor = (current - ACCEPT_ZNCC_TOLERANCE).max(start_score);
        if next_score.is_nan() || next_score < floor {
            return end(state, measured, iterations, LoopStop::Rejected);
        }
        if largest_move >= previous_move {
            return if next_score >= current {
                end(next, next_measured, iterations, LoopStop::Oscillation)
            } else {
                end(state, measured, iterations, LoopStop::Oscillation)
            };
        }
        (state, measured, current) = (next, next_measured, next_score);
        previous_move = largest_move;
    }
}

#[cfg(test)]
mod tests;
