// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

/// The viewer's own binary allocates through `mimalloc`, as the Python
/// extension that `sfm explorer` runs the viewer from does, so the two builds
/// of the viewer behave the same. The library sets no allocator: that is the
/// final artifact's choice. See `specs/gui/architecture.md` § "Key Rust
/// Crates".
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    sfm_explorer::run();
}
