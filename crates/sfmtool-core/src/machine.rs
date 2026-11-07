// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Facts about the machine that decide how sfmtool runs on it: the SIMD level
//! the kernels dispatch to, the CPU, the thread count of the rayon pool, the
//! physical memory and the GPU adapter the GPU code would get.
//!
//! `sfm version` prints these for bug reports
//! (`specs/cli/other/version-command.md`), through the extension's internal
//! `_machine_facts`. Each function reads the fact the way the code that
//! depends on it does, so the report says what a run would do rather than
//! what the hardware could do.

use std::time::Duration;

use crate::progress::{Cancelled, Progress};

/// The widest SIMD instruction set the Rust kernels dispatch to on this CPU.
///
/// The kernels choose a path at run time with `is_x86_feature_detected!`, one
/// call at each dispatch site, and fall back to a narrower path or to scalar
/// code when a feature is missing. Their requirements come in three tiers:
/// SSE2 (part of every x86-64 CPU), AVX2 alone (the focal vote's residuals,
/// homography estimation, the KD-forest distances, the anisotropic remap) and
/// AVX2 with FMA (the SIFT scale space, orientation and descriptor, and the
/// patch kernels). There is no AVX-512 code, and no SIMD code for any other
/// architecture, whose kernels are all scalar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdLevel {
    /// No SIMD path: a target other than x86-64.
    Scalar,
    /// x86-64 without AVX2: the SSE2 paths, and scalar code where a kernel
    /// has no SSE2 path.
    Sse2,
    /// AVX2 without FMA: the AVX2-only kernels take their AVX2 path, and the
    /// AVX2+FMA kernels fall back.
    Avx2,
    /// AVX2 and FMA: every SIMD path.
    Avx2Fma,
}

impl SimdLevel {
    /// The label `sfm version` prints: `AVX2+FMA`, `AVX2`, `SSE2` or
    /// `none (scalar)`.
    pub fn label(self) -> &'static str {
        match self {
            SimdLevel::Scalar => "none (scalar)",
            SimdLevel::Sse2 => "SSE2",
            SimdLevel::Avx2 => "AVX2",
            SimdLevel::Avx2Fma => "AVX2+FMA",
        }
    }

    /// The level for an x86-64 CPU with or without AVX2 and FMA.
    fn x86_64(avx2: bool, fma: bool) -> SimdLevel {
        match (avx2, fma) {
            (true, true) => SimdLevel::Avx2Fma,
            (true, false) => SimdLevel::Avx2,
            (false, _) => SimdLevel::Sse2,
        }
    }
}

/// The SIMD level the kernels dispatch to on this CPU, detected with the same
/// `is_x86_feature_detected!` checks the kernels make.
///
/// The diagnostic variables `SFMTOOL_SIFT_NO_AVX2` and
/// `SFMTOOL_FOCAL_VOTE_NO_SIMD`, which force scalar paths in single kernels,
/// are not reflected.
pub fn simd_level() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        SimdLevel::x86_64(
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("fma"),
        )
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        SimdLevel::Scalar
    }
}

/// The CPU's model name, such as `"Intel(R) Core(TM) i9-14900HX"`, or `None`
/// where it cannot be read cheaply.
///
/// It is the processor brand string of the `cpuid` instruction on x86-64, and
/// the `machdep.cpu.brand_string` sysctl on other macOS targets (Apple
/// silicon). Other targets return `None`.
pub fn cpu_model() -> Option<String> {
    #[cfg(target_arch = "x86_64")]
    {
        x86_64_brand_string()
    }
    #[cfg(all(not(target_arch = "x86_64"), target_os = "macos"))]
    {
        macos_brand_string()
    }
    #[cfg(not(any(target_arch = "x86_64", target_os = "macos")))]
    {
        None
    }
}

#[cfg(target_arch = "x86_64")]
fn x86_64_brand_string() -> Option<String> {
    use std::arch::x86_64::__cpuid;
    // Leaf 0x8000_0000 gives the highest extended leaf; the brand string is
    // the 48 bytes of leaves 0x8000_0002 to 0x8000_0004, NUL-padded.
    #[allow(unused_unsafe)]
    // SAFETY: `cpuid` is present on every x86-64 CPU and only reads registers.
    let max_leaf = unsafe { __cpuid(0x8000_0000) }.eax;
    if max_leaf < 0x8000_0004 {
        return None;
    }
    let mut bytes = Vec::with_capacity(48);
    for leaf in 0x8000_0002u32..=0x8000_0004 {
        #[allow(unused_unsafe)]
        // SAFETY: as above, and the leaf is within the reported range.
        let r = unsafe { __cpuid(leaf) };
        for word in [r.eax, r.ebx, r.ecx, r.edx] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    }
    clean_model_name(&bytes)
}

#[cfg(all(not(target_arch = "x86_64"), target_os = "macos"))]
fn macos_brand_string() -> Option<String> {
    let name = c"machdep.cpu.brand_string";
    let mut buf = [0u8; 256];
    let mut len = buf.len();
    // SAFETY: `name` is NUL-terminated, `buf` is writable for `len` bytes, and
    // sysctlbyname writes at most `len` bytes and updates `len`.
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            buf.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 {
        return None;
    }
    clean_model_name(&buf[..len.min(buf.len())])
}

/// The model name in `bytes`, up to the first NUL, with surrounding spaces
/// removed and runs of spaces inside collapsed (Intel pads the brand string
/// with leading spaces); `None` when nothing is left or it is not UTF-8.
#[cfg_attr(
    not(any(target_arch = "x86_64", target_os = "macos")),
    allow(dead_code)
)]
fn clean_model_name(bytes: &[u8]) -> Option<String> {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let text = std::str::from_utf8(&bytes[..end]).ok()?;
    let name = text.split_whitespace().collect::<Vec<_>>().join(" ");
    (!name.is_empty()).then_some(name)
}

/// The number of logical CPUs this process may run on, as
/// `std::thread::available_parallelism` gives it (it counts an affinity mask
/// or a cgroup quota where the OS applies one), or `None` when it cannot be
/// read.
pub fn logical_cores() -> Option<usize> {
    std::thread::available_parallelism().ok().map(|n| n.get())
}

/// The number of threads the parallel kernels run on: that of the rayon pool
/// the caller is in, which outside any pool is the global pool. The global
/// pool takes `RAYON_NUM_THREADS` when it is set to a positive number, and
/// otherwise one thread per logical CPU. Reading this starts the global pool
/// if no kernel has yet.
pub fn rayon_threads() -> usize {
    rayon::current_num_threads()
}

/// The machine's physical memory in bytes, or `None` where it cannot be read.
///
/// It is read with `GlobalMemoryStatusEx` on Windows and with
/// `sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE)` on Linux and macOS.
/// Other targets return `None`.
pub fn physical_memory_bytes() -> Option<u64> {
    os_physical_memory_bytes()
}

#[cfg(windows)]
fn os_physical_memory_bytes() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: MEMORYSTATUSEX is plain data, so all zeroes is a valid value, and
    // GlobalMemoryStatusEx only writes into the struct it is handed, whose
    // `dwLength` it requires to be set to its size.
    unsafe {
        let mut status: MEMORYSTATUSEX = std::mem::zeroed();
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        if GlobalMemoryStatusEx(&mut status) != 0 && status.ullTotalPhys > 0 {
            Some(status.ullTotalPhys)
        } else {
            None
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn os_physical_memory_bytes() -> Option<u64> {
    // SAFETY: sysconf takes an integer name and returns an integer; it touches
    // no memory of ours.
    let (pages, page_size) = unsafe {
        (
            libc::sysconf(libc::_SC_PHYS_PAGES),
            libc::sysconf(libc::_SC_PAGESIZE),
        )
    };
    if pages > 0 && page_size > 0 {
        (pages as u64).checked_mul(page_size as u64)
    } else {
        None
    }
}

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
fn os_physical_memory_bytes() -> Option<u64> {
    None
}

/// The kind of GPU adapter, from wgpu's `DeviceType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuDeviceType {
    /// A GPU of its own.
    Discrete,
    /// A GPU that shares the CPU's memory.
    Integrated,
    /// A GPU of a virtual machine.
    Virtual,
    /// A software rasterizer that runs on the CPU, such as lavapipe or WARP.
    Cpu,
    /// A kind the driver does not name.
    Other,
}

impl GpuDeviceType {
    /// The label `sfm version` prints: `discrete`, `integrated`, `virtual`,
    /// `CPU (software)` or `other`.
    pub fn label(self) -> &'static str {
        match self {
            GpuDeviceType::Discrete => "discrete",
            GpuDeviceType::Integrated => "integrated",
            GpuDeviceType::Virtual => "virtual",
            GpuDeviceType::Cpu => "CPU (software)",
            GpuDeviceType::Other => "other",
        }
    }
}

/// The GPU adapter a run would get: what the driver reports for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuAdapter {
    /// The adapter's name, such as `"NVIDIA GeForce RTX 4090 Laptop GPU"`.
    pub name: String,
    /// The graphics API wgpu reaches it through: `dx12`, `vulkan`, `metal` or
    /// `gl`.
    pub backend: String,
    /// The kind of adapter.
    pub device_type: GpuDeviceType,
}

/// What [`probe_gpu_adapter`] found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuProbe {
    /// The adapter the request returned.
    Found(GpuAdapter),
    /// The request returned no adapter; the text is wgpu's reason.
    NoAdapter(String),
    /// The request did not return within the timeout. The thread making it is
    /// left running, since a driver call cannot be interrupted.
    TimedOut(Duration),
    /// The request panicked inside wgpu or the driver.
    Failed,
    /// `sfmtool-core` was built without its `gpu` feature, so no GPU code runs.
    NotBuilt,
}

/// Which GPU adapter the GPU code would run on: the request
/// `GpuContext::new` makes (every backend compiled in, the high-performance
/// adapter, no forced software fallback), made on a thread of its own and
/// waited for at most `timeout`, so a driver that hangs cannot hang the
/// caller. A `timeout` too long to add to the current instant, such as
/// [`Duration::MAX`], means waiting with no deadline.
///
/// It never panics: a panic inside wgpu or the driver is reported as
/// [`GpuProbe::Failed`]. The viewer requests its adapter with its window's
/// surface and its platform's one backend, so it can get a different one.
///
/// # Errors
///
/// [`Cancelled`] when `progress` is cancelled while waiting; the request goes
/// on in its thread, as after a timeout.
pub fn probe_gpu_adapter(timeout: Duration, progress: &Progress) -> Result<GpuProbe, Cancelled> {
    #[cfg(feature = "gpu")]
    {
        let request = || {
            pollster::block_on(crate::features::optical_flow::gpu::request_compute_adapter())
                .map(|adapter| gpu_adapter(&adapter.get_info()))
                .map_err(|error| error.to_string())
        };
        wait_for(request, timeout, progress)
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = timeout;
        progress.check_cancel()?;
        Ok(GpuProbe::NotBuilt)
    }
}

#[cfg(feature = "gpu")]
fn gpu_adapter(info: &wgpu::AdapterInfo) -> GpuAdapter {
    GpuAdapter {
        name: info.name.clone(),
        backend: info.backend.to_str().to_owned(),
        device_type: match info.device_type {
            wgpu::DeviceType::DiscreteGpu => GpuDeviceType::Discrete,
            wgpu::DeviceType::IntegratedGpu => GpuDeviceType::Integrated,
            wgpu::DeviceType::VirtualGpu => GpuDeviceType::Virtual,
            wgpu::DeviceType::Cpu => GpuDeviceType::Cpu,
            wgpu::DeviceType::Other => GpuDeviceType::Other,
        },
    }
}

/// Run `request` on a thread of its own and wait for its answer for at most
/// `timeout`, checking `progress` for cancellation as it waits.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
fn wait_for<F>(request: F, timeout: Duration, progress: &Progress) -> Result<GpuProbe, Cancelled>
where
    F: FnOnce() -> Result<GpuAdapter, String> + Send + 'static,
{
    use std::sync::mpsc::{channel, RecvTimeoutError};
    use std::time::Instant;

    // How often the wait checks for cancellation.
    const POLL: Duration = Duration::from_millis(50);

    let (sender, receiver) = channel();
    let spawned = std::thread::Builder::new()
        .name("gpu-adapter-probe".into())
        .spawn(move || {
            // The receiver is gone after a timeout; nothing is waiting then.
            let _ = sender.send(request());
        });
    if spawned.is_err() {
        return Ok(GpuProbe::Failed);
    }
    // A timeout too long to add to the current instant waits with no deadline.
    let deadline = Instant::now().checked_add(timeout);
    loop {
        progress.check_cancel()?;
        let wait = match deadline {
            Some(deadline) => {
                let now = Instant::now();
                if now >= deadline {
                    return Ok(GpuProbe::TimedOut(timeout));
                }
                POLL.min(deadline - now)
            }
            None => POLL,
        };
        match receiver.recv_timeout(wait) {
            Ok(Ok(adapter)) => return Ok(GpuProbe::Found(adapter)),
            Ok(Err(reason)) => return Ok(GpuProbe::NoAdapter(reason)),
            // The thread ended without sending: the request panicked.
            Err(RecvTimeoutError::Disconnected) => return Ok(GpuProbe::Failed),
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
}

#[cfg(test)]
mod tests;
