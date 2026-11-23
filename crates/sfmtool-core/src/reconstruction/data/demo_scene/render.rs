// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The demo scene's geometry and its ray tracer: the five camera poses, the
//! ray's hit on the cube (the slab test) and on the ground, and the
//! supersampled render of one photograph.
//!
//! The geometry is `f64`: the scene's coordinates reach 10 m and a texture
//! coordinate has to resolve a fraction of a 2 cm cell, about `10⁻⁴` of the
//! coordinate's size, which `f32`'s `6 · 10⁻⁸` holds but leaves no margin
//! through the slab test's divisions. The samples of a pixel are averaged in
//! `f32`: at most a few hundred 8-bit samples, whose `f32` sum keeps each
//! pixel to about `10⁻⁵` of a level before it is rounded to 8 bits.

use crate::camera::CameraIntrinsics;

use super::texture::{Surface, CUBE_A, CUBE_B};

/// The colour a ray that meets no surface sees, RGB.
pub(super) const BACKGROUND: [u8; 3] = [60, 70, 80];

/// The half side of the ground square `[-5, 5] × [-5, 5]`, in metres.
pub(super) const GROUND_HALF_SIDE: f64 = 5.0;

/// The point every camera looks at: the cube's centre.
const TARGET: [f64; 3] = [0.0, 0.0, 0.5];

/// The camera centres: on an arc 4 m from the `z` axis on the cube's `-y`
/// side, at azimuths −55°, −27.5°, 0°, 27.5° and 55° from `-y`, alternating
/// between the heights 1.2 m and 3.0 m. Written as decimal literals rather
/// than computed with `sin` and `cos`, so the poses are built from `+`, `−`,
/// `×`, `÷` and `sqrt` alone, which IEEE 754 rounds the same on every
/// platform.
pub(super) const CAMERA_CENTRES: [[f64; 3]; 5] = [
    [-3.277, -2.294, 1.2],
    [-1.847, -3.548, 3.0],
    [0.0, -4.0, 1.2],
    [1.847, -3.548, 3.0],
    [3.277, -2.294, 1.2],
];

pub(super) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let length = dot(a, a).sqrt();
    a.map(|c| c / length)
}

/// A camera's world-to-camera rotation, as its rows: the camera's right, up
/// and back axes in world coordinates, so `p_cam = R · (p_world − C)` with the
/// camera looking down `−Z` with `+Y` up. The camera looks at [`TARGET`] with
/// its `+Y` axis in the vertical plane through its line of sight, upward.
pub(super) fn camera_rotation(centre: [f64; 3]) -> [[f64; 3]; 3] {
    let forward = normalize([
        TARGET[0] - centre[0],
        TARGET[1] - centre[1],
        TARGET[2] - centre[2],
    ]);
    let right = normalize(cross(forward, [0.0, 0.0, 1.0]));
    let up = cross(right, forward);
    [right, up, forward.map(|c| -c)]
}

/// Where a ray from `origin` along `direction` first meets a surface: the
/// surface and the distance along `direction`, or `None` for the background.
pub(super) fn trace(origin: [f64; 3], direction: [f64; 3]) -> Option<(Surface, f64)> {
    let cube = cube_hit(origin, direction);
    let ground = ground_hit(origin, direction);
    match (cube, ground) {
        (Some(c), Some(g)) => Some(if c.1 <= g.1 { c } else { g }),
        (c, g) => c.or(g),
    }
}

/// The ray's hit on the ground square, seen from above: `λ = −C_z / d_z`
/// where `d_z < 0`, when the hit lies inside the square.
fn ground_hit(origin: [f64; 3], direction: [f64; 3]) -> Option<(Surface, f64)> {
    if !(direction[2] < 0.0 && origin[2] > 0.0) {
        return None;
    }
    let lambda = -origin[2] / direction[2];
    let x = origin[0] + lambda * direction[0];
    let y = origin[1] + lambda * direction[1];
    (x.abs() <= GROUND_HALF_SIDE && y.abs() <= GROUND_HALF_SIDE)
        .then_some((Surface::Ground, lambda))
}

/// The ray's entry into the cube, by the slab test in the cube's own axes
/// `a`, `b`, `z`: the origin and direction are expressed in those axes by dot
/// products, and the entry distance is the largest of the three slabs' near
/// distances when it is below the smallest far distance and positive. The
/// surface is the face of the slab whose near distance is the entry.
pub(super) fn cube_hit(origin: [f64; 3], direction: [f64; 3]) -> Option<(Surface, f64)> {
    const Z: [f64; 3] = [0.0, 0.0, 1.0];
    // Per axis: the slab's bounds, and the faces entered moving up and down it.
    let slabs = [
        (CUBE_A, -0.5, 0.5, Surface::MinusA, Surface::PlusA),
        (CUBE_B, -0.5, 0.5, Surface::MinusB, Surface::PlusB),
        (Z, 0.0, 1.0, Surface::Base, Surface::Top),
    ];
    let mut entry = f64::NEG_INFINITY;
    let mut exit = f64::INFINITY;
    let mut entry_face = None;
    for (axis, low, high, low_face, high_face) in slabs {
        let o = dot(origin, axis);
        let d = dot(direction, axis);
        if d == 0.0 {
            if o < low || o > high {
                return None;
            }
            continue;
        }
        let (near, far, face) = if d > 0.0 {
            ((low - o) / d, (high - o) / d, low_face)
        } else {
            ((high - o) / d, (low - o) / d, high_face)
        };
        if near > entry {
            entry = near;
            entry_face = Some(face);
        }
        exit = exit.min(far);
    }
    match entry_face {
        Some(face) if entry < exit && entry > 0.0 => Some((face, entry)),
        _ => None,
    }
}

/// The world direction of the ray through the pixel position `(u, v)` of a
/// camera with intrinsics `camera` and world-to-camera rotation `rotation`:
/// `Rᵀ · pixel_to_ray(u, v)`.
pub(super) fn pixel_ray(
    camera: &CameraIntrinsics,
    rotation: &[[f64; 3]; 3],
    u: f64,
    v: f64,
) -> [f64; 3] {
    let ray = camera.pixel_to_ray(u, v);
    let [r, up, back] = rotation;
    [
        ray[0] * r[0] + ray[1] * up[0] + ray[2] * back[0],
        ray[0] * r[1] + ray[1] * up[1] + ray[2] * back[1],
        ray[0] * r[2] + ray[1] * up[2] + ray[2] * back[2],
    ]
}

/// The colour a ray sees, RGB on `[0, 255]` unrounded.
pub(super) fn ray_colour(origin: [f64; 3], direction: [f64; 3], seed: u64) -> [f32; 3] {
    match trace(origin, direction) {
        Some((surface, lambda)) => {
            let x = [
                origin[0] + lambda * direction[0],
                origin[1] + lambda * direction[1],
                origin[2] + lambda * direction[2],
            ];
            surface.colour(x, seed)
        }
        None => BACKGROUND.map(f32::from),
    }
}

/// Render row `row` of a photograph into `out`, RGB: each pixel is the mean
/// of `supersampling²` rays through a regular grid of sample positions inside
/// it, at offsets `(i + 0.5) / supersampling` along each axis, summed in a
/// fixed order and rounded to 8 bits.
pub(super) fn render_row(
    camera: &CameraIntrinsics,
    centre: [f64; 3],
    rotation: &[[f64; 3]; 3],
    supersampling: u32,
    seed: u64,
    row: u32,
    out: &mut [u8],
) {
    let s = f64::from(supersampling);
    let samples = (u64::from(supersampling) * u64::from(supersampling)) as f32;
    for (col, pixel) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let mut sum = [0.0f32; 3];
        for j in 0..supersampling {
            let v = f64::from(row) + (f64::from(j) + 0.5) / s;
            for i in 0..supersampling {
                let u = col as f64 + (f64::from(i) + 0.5) / s;
                let colour = ray_colour(centre, pixel_ray(camera, rotation, u, v), seed);
                for c in 0..3 {
                    sum[c] += colour[c];
                }
            }
        }
        for c in 0..3 {
            pixel[c] = (sum[c] / samples).round().clamp(0.0, 255.0) as u8;
        }
    }
}
