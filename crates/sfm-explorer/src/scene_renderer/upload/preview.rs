// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The camera model proposal's **preview buffers**: a target per row of a
//! node's point buffers and per slot of its surfel buffers, which the preview
//! variants of the point and patch pipelines blend toward by the node's
//! `preview_s`.
//!
//! See `specs/gui/point-cloud-rendering.md` § "The camera model proposal's
//! preview" and `specs/gui/patch-rendering.md`. The rows are built on the
//! proposal job's thread, one per point buffer row
//! ([`crate::camera_model_proposal::preview::PreviewTargets`]),
//! so a new answer is one upload of each buffer here, and a frame of motion
//! writes nothing but the blend into the node's uniforms. The base's point and
//! surfel buffers are never written, so ending the preview drops these and
//! restores nothing.

use std::collections::HashMap;

use wgpu::util::DeviceExt;

use super::super::gpu_types::SurfelPreview;
use super::super::recon::PreviewResources;
use super::super::SceneRenderer;
use crate::camera_model_proposal::preview::{PreviewTargets, FLAG_STILL};
use crate::camera_model_proposal::PreviewDraw;

impl SceneRenderer {
    /// Hold the preview `draw` describes on its node, and none on any other.
    ///
    /// Called once per frame after the uploads. A generation the node's
    /// buffers were not written for is one upload; the same generation only
    /// moves the blend. A node whose point buffers no longer have the rows the
    /// targets were laid out for (a new base, a changed addition set) gets no
    /// preview rather than one that moves the wrong points.
    pub(crate) fn set_preview(&mut self, device: &wgpu::Device, draw: Option<&PreviewDraw>) {
        for (&id, bundle) in self.recons.iter_mut() {
            let Some(draw) = draw.filter(|draw| draw.recon == id) else {
                bundle.preview = None;
                continue;
            };
            let targets = &draw.targets;
            let fits = targets.base.len() == bundle.point_count as usize
                && targets.additions.len() == bundle.addition_count() as usize;
            if !fits {
                bundle.preview = None;
                continue;
            }
            match bundle.preview.as_mut() {
                Some(preview) if preview.generation == targets.generation => preview.s = draw.s,
                _ => {
                    let surfel_rows = |patch: Option<&super::super::recon::PatchResources>| {
                        patch.map(|patch| {
                            surfel_preview_rows(&patch.slot_of_point, patch.count, targets)
                        })
                    };
                    let patches = surfel_rows(bundle.patch.as_ref());
                    let addition_patches = surfel_rows(
                        bundle
                            .additions
                            .as_ref()
                            .and_then(|additions| additions.patch.as_ref()),
                    );
                    let vec4_buffer = |label: &str, rows: &[[f32; 4]]| {
                        let still = [[0.0, 0.0, 0.0, FLAG_STILL]];
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(label),
                            // At least one row, so a point-free node still has
                            // a bindable buffer.
                            contents: bytemuck::cast_slice(if rows.is_empty() {
                                &still
                            } else {
                                rows
                            }),
                            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        })
                    };
                    let surfel_buffer = |label: &str, rows: &[SurfelPreview]| {
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(label),
                            contents: bytemuck::cast_slice(rows),
                            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        })
                    };
                    bundle.preview = Some(PreviewResources {
                        generation: targets.generation,
                        s: draw.s,
                        points: vec4_buffer("point preview", &targets.base),
                        additions: (!targets.additions.is_empty())
                            .then(|| vec4_buffer("addition point preview", &targets.additions)),
                        patches: patches
                            .as_deref()
                            .map(|rows| surfel_buffer("surfel preview", rows)),
                        addition_patches: addition_patches
                            .as_deref()
                            .map(|rows| surfel_buffer("addition surfel preview", rows)),
                        #[cfg(test)]
                        built_patches: patches.unwrap_or_default(),
                    });
                }
            }
        }
    }
}

/// The surfel preview rows for one surfel buffer, by slot: each slot takes the
/// row of the point it was packed from, through `slot_of_point`, the map the
/// deleted mask's surfel half is written through. A slot whose point the
/// preview does not move keeps the still flag and a size factor of 1.
///
/// `slot_of_point` is keyed by the version's index, a base row below the base
/// buffer's row count and an addition's row past it, as the instances carry it.
pub(super) fn surfel_preview_rows(
    slot_of_point: &HashMap<u32, u32>,
    count: u32,
    targets: &PreviewTargets,
) -> Vec<SurfelPreview> {
    let still = SurfelPreview {
        target: [0.0, 0.0, 0.0, FLAG_STILL],
        size_factor: 1.0,
    };
    let mut rows = vec![still; count.max(1) as usize];
    let base_count = targets.base.len();
    for (&point, &slot) in slot_of_point {
        let point = point as usize;
        let row = if point < base_count {
            targets.base.get(point).zip(targets.base_size.get(point))
        } else {
            let k = point - base_count;
            targets.additions.get(k).zip(targets.additions_size.get(k))
        };
        if let (Some(entry), Some((target, size_factor))) = (rows.get_mut(slot as usize), row) {
            *entry = SurfelPreview {
                target: *target,
                size_factor: *size_factor,
            };
        }
    }
    rows
}
