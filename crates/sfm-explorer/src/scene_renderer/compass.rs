// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Procedural mesh geometry for the orientation compass indicator.
//!
//! Generates the wireframe edges (vertical axis + horizontal ring) and the
//! filled 8-point star rose consumed by the target-indicator pipeline. The GPU
//! layout type ([`CompassEdgeInstance`]) lives in [`super::gpu_types`]; this
//! module only produces the vertex data.

use super::gpu_types::CompassEdgeInstance;

/// Number of segments for the horizontal ring circle.
const RING_SEGMENTS: usize = 32;

/// Radius of the horizontal ring circle and star outer tips.
const RING_RADIUS: f32 = 0.6;

/// Number of star points (cardinal + intercardinal directions).
const STAR_POINTS: usize = 8;

/// Inner circle radius for the star indentations (1/5 of outer radius).
const STAR_INNER_RADIUS: f32 = RING_RADIUS / 5.0;

/// Outer tip radii: cardinal (N/S/E/W) tips extend beyond the ring,
/// intercardinal tips are shorter.
const STAR_CARDINAL_RADIUS: f32 = RING_RADIUS * 1.25;
const STAR_INTERCARDINAL_RADIUS: f32 = RING_RADIUS * 0.8;

/// Wireframe edges: vertical axis + circular ring.
///
/// The horizontal compass rose is rendered as a filled star polygon, so only
/// the vertical spikes and ring remain as wireframe edges.
pub(super) fn create_compass_edge_instances() -> Vec<CompassEdgeInstance> {
    let mut edges: Vec<CompassEdgeInstance> = Vec::new();

    // Vertical axis: center to top, center to bottom
    let center = [0.0, 0.0, 0.0, 1.0];
    edges.push(CompassEdgeInstance {
        endpoint_a: center,
        endpoint_b: [0.0, 0.0, 1.5, 1.0], // top spike
    });
    edges.push(CompassEdgeInstance {
        endpoint_a: center,
        endpoint_b: [0.0, 0.0, -0.7, 1.0], // bottom spike
    });

    // Circular ring in the horizontal (z=0) plane
    for i in 0..RING_SEGMENTS {
        let angle_a = (i as f32) * std::f32::consts::TAU / RING_SEGMENTS as f32;
        let angle_b =
            ((i + 1) % RING_SEGMENTS) as f32 * std::f32::consts::TAU / RING_SEGMENTS as f32;
        edges.push(CompassEdgeInstance {
            endpoint_a: [
                RING_RADIUS * angle_a.cos(),
                RING_RADIUS * angle_a.sin(),
                0.0,
                1.0,
            ],
            endpoint_b: [
                RING_RADIUS * angle_b.cos(),
                RING_RADIUS * angle_b.sin(),
                0.0,
                1.0,
            ],
        });
    }

    edges
}

/// Generate a filled 8-point star polygon as a triangle list in the z=0 plane.
///
/// Cardinal tips (N/E/S/W, indices 0/2/4/6) extend to `STAR_CARDINAL_RADIUS`.
/// Intercardinal tips (NE/SE/SW/NW, indices 1/3/5/7) extend to
/// `STAR_INTERCARDINAL_RADIUS`. Inner indentations are offset 2/3 towards
/// the intercardinal tip, giving cardinals wider bases and intercardinals
/// narrower bases. Triangulated as a center fan (16 triangles).
pub(super) fn create_compass_star_mesh() -> Vec<[f32; 3]> {
    use std::f32::consts::TAU;

    let tip_radius = |i: usize| -> f32 {
        if i.is_multiple_of(2) {
            STAR_CARDINAL_RADIUS
        } else {
            STAR_INTERCARDINAL_RADIUS
        }
    };

    let mut verts = Vec::with_capacity(STAR_POINTS * 2 * 3);
    let center = [0.0f32, 0.0, 0.0];

    for i in 0..STAR_POINTS {
        let outer_angle = i as f32 * TAU / STAR_POINTS as f32;
        // Offset inner vertex 2/3 towards the intercardinal side:
        // even i (cardinal→intercardinal): 2/3 towards next
        // odd i (intercardinal→cardinal): 1/3 towards next (= 2/3 back towards current)
        let inner_offset = if i.is_multiple_of(2) {
            2.0 / 3.0
        } else {
            1.0 / 3.0
        };
        let inner_angle = (i as f32 + inner_offset) * TAU / STAR_POINTS as f32;
        let next_i = (i + 1) % STAR_POINTS;
        let next_outer_angle = next_i as f32 * TAU / STAR_POINTS as f32;

        let r = tip_radius(i);
        let outer = [r * outer_angle.cos(), r * outer_angle.sin(), 0.0];
        let inner = [
            STAR_INNER_RADIUS * inner_angle.cos(),
            STAR_INNER_RADIUS * inner_angle.sin(),
            0.0,
        ];
        let r_next = tip_radius(next_i);
        let next_outer = [
            r_next * next_outer_angle.cos(),
            r_next * next_outer_angle.sin(),
            0.0,
        ];

        // Triangle: center → outer tip → inner indentation
        verts.extend_from_slice(&[center, outer, inner]);
        // Triangle: center → inner indentation → next outer tip
        verts.extend_from_slice(&[center, inner, next_outer]);
    }

    verts
}
