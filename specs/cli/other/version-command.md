# `sfm version` Command

`sfm version` prints the version of the `sfmtool` package and, when the code
running is not the release that version names, which commit it comes from.
After that line it prints a short report of the facts that decide how
sfmtool runs on this machine, for bug reports and for agents: the build, the
install, Python, the OS, the CPU, the thread count, the SIMD level, memory,
the GPU adapter, pycolmap and its CUDA support, numpy, OpenCV and ffmpeg.
The package version in `pyproject.toml` and `crates/sfmtool-py/Cargo.toml`
changes only at a release, so every build of a later commit carries the
version of the last release; the commit is what tells such a build from the
release. The version itself is unchanged: this is display only, and
`importlib.metadata.version("sfmtool")` still reads the package metadata.
Nothing in the output depends on the current directory: no workspace is
looked for.

## Command Syntax

```bash
sfm version [--bare] [--json]
```

| Option | Effect |
|--------|--------|
| `--bare` | Print only the version line. It reads no other fact: it imports neither pycolmap, numpy nor OpenCV and makes no GPU request, so it takes the time of `import sfmtool` and two git commands. |
| `--json` | Print the same facts as one JSON object (see [JSON](#json)). With `--bare`, the object holds only its `sfmtool` key, which is no slower than `--bare`. |

## Output

```text
sfmtool 0.3 (v0.3-388-ge10b06192-dirty)
Build:    release, extension built from v0.3-388-ge10b06192-dirty, checkout v0.3-388-ge10b06192-dirty
Install:  editable, C:\Dev\prod\sfmtool\src\sfmtool
Python:   3.14.9, C:\Dev\prod\sfmtool\.pixi\envs\default\python.exe
OS:       Windows-11-10.0.26300-SP0, AMD64
CPU:      Intel(R) Core(TM) i9-14900HX, 32 logical cores
Threads:  32
SIMD:     AVX2+FMA
Memory:   63.7 GiB
GPU:      NVIDIA GeForce RTX 4090 Laptop GPU (vulkan, discrete)
pycolmap: 4.2.1, COLMAP 4.2.1, CPU only
numpy:    2.5.4
OpenCV:   5.0.0
ffmpeg:   9.0.2 (C:\Dev\prod\sfmtool\.pixi\envs\default\Library\bin\ffmpeg.EXE)
```

The first line is the version line, described next, and is all that
`--bare` prints. Each later line is one fact, its label padded to ten
columns; [The report](#the-report) gives each one's meaning and source. A
fact that cannot be read says so in its line, and the command still exits
with status 0.

## The version line

The line starts `sfmtool <version>`, the package metadata version (`unknown`
when the package metadata is not installed). What follows it depends on two
`git describe` strings, each the output of
`git describe --tags --long --always` with `-dirty` added when
`git status --porcelain --untracked-files=no` lists a change:

- **the checkout's**, run when the command runs, in the git checkout the
  `sfmtool` package is run from (see below);
- **the extension's**, recorded when the compiled extension
  `sfmtool._sfmtool` was built, in the source tree it was built from, and read
  through the extension's internal `_build_git_describe()`.

Either may be unknown. Its form is `<tag>-<n>-g<hash>`: the nearest tag, the
number of commits since it and the commit's abbreviated hash, with `-dirty`
added when a tracked file differed from that commit, staged or not
(untracked files do not count). This is what `git describe --dirty` prints;
it is computed with `git status` because `git describe --dirty` rewrites the
git index when the index's file times are out of date, so printing the
version or building the extension would write to the repository, while
`git status` run with `GIT_OPTIONAL_LOCKS=0` leaves it unchanged. With no tag
reachable, as in a shallow clone, it is the abbreviated hash alone (that is
what `--always` adds).

| Checkout | Extension | Printed |
|----------|-----------|---------|
| unknown | unknown | `sfmtool 0.3` |
| `v0.3-0-g276e8d986`, or unknown | `v0.3-0-g276e8d986` | `sfmtool 0.3` |
| `v0.3-385-g276e8d986` | `v0.3-385-g276e8d986`, or unknown | `sfmtool 0.3 (v0.3-385-g276e8d986)` |
| `v0.3-385-g276e8d986-dirty` | `v0.3-385-g276e8d986-dirty` | `sfmtool 0.3 (v0.3-385-g276e8d986-dirty)` |
| `v0.3-386-g1a2b3c4d5` | `v0.3-385-g276e8d986` | `sfmtool 0.3 (checkout v0.3-386-g1a2b3c4d5, extension built from v0.3-385-g276e8d986)` |

In words:

- **Both known and different**: both are printed in parentheses, labelled
  `checkout` and `extension built from`. This is an extension that was not
  rebuilt after the checkout moved to another commit or after its Rust sources
  changed, a stale `maturin develop`; the line shows it whether or not either
  string is a release.
- **Otherwise**, the one string known (the two being equal when both are) is
  printed in parentheses after the version, unless it names the release: a
  clean tree exactly at the tag `v<version>` or `<version>`, that is
  `v0.3-0-g<hash>` for version `0.3`. A tag that is not the package version,
  such as `v0.4-0-g<hash>` while the package says `0.3`, is printed.
- **Neither known**: the version alone.

So a wheel built at its release tag prints `sfmtool 0.3`, and so does an
editable checkout exactly at that tag, clean, with an extension built there.

## Where the checkout's describe comes from

The package is in a checkout when its directory is `<root>/src/sfmtool` and
`<root>` holds `.git` (a directory, or the file of a linked worktree), as in
the editable install `pixi` makes. The command then runs
`git -C <root> describe --tags --long --always` and
`git -C <root> status --porcelain --untracked-files=no`, each with a
two-second timeout. It skips the describe, with no message, when `<root>` has no `.git`
(an install into site-packages, including one in a virtual environment inside
some other repository, whose `<root>` is not that repository's root), when git
is not installed, when either git command exits with an error, and when one
runs past the timeout. Git runs with `GIT_OPTIONAL_LOCKS=0`, and with `GIT_DIR` and
`GIT_WORK_TREE` removed from its environment so that `-C <root>` decides the
repository. This is [`checkout_describe`](../../../src/sfmtool/_version.py),
and the line is formatted by `version_text` in the same file.

## Where the extension's describe comes from

[`build.rs`](../../../crates/sfmtool-py/build.rs) of the `sfmtool-py` crate
runs the same two commands in the repository root two directories above the crate,
when that root holds `.git`, and records its output in the compile-time
variable `SFMTOOL_BUILD_GIT_DESCRIBE`, which `_build_git_describe()` in
[`lib.rs`](../../../crates/sfmtool-py/src/lib.rs) returns (`None` for the
empty string). It records the empty string, and the build goes on, when there
is no `.git` (a build from the sdist, which carries none), when git is not
installed, when git fails, and when git runs past a ten-second timeout. It
uses no crate beyond the standard library.

Cargo runs the script again, and so recompiles `sfmtool-py` with the new
string, when one of these changes:

- the git `HEAD` file, the ref of the branch it names, `packed-refs` and the
  `refs/tags` directory, so a commit, a checkout or a new tag is seen;
- each crate's `Cargo.toml`, `build.rs` and `src/` directory, and the
  workspace's `Cargo.toml` and `Cargo.lock`, so `-dirty` follows an edit to
  any source the extension is compiled from, staged or not.

The git index is not watched: `git add`, and a plain `git status` whenever
file times are out of date, rewrite it without changing the describe, and each
run of the script recompiles `sfmtool-py`. An edit to any other file, such as
a Python module, does not run it again by itself, so the recorded string
marks such an edit `-dirty` only from the next time the script runs. Only
paths that exist are watched, since Cargo runs a build script on every build
while a path it watches is missing.

## The report

[`_machine_report.py`](../../../src/sfmtool/_machine_report.py) collects
the report (`collect_report`) and formats it (`report_text`). The facts read
in Rust come from the extension's internal `_machine_facts(gpu_timeout_s)`,
which calls the functions of
[`sfmtool_core::machine`](../../../crates/sfmtool-core/src/machine.rs), so
each is read the way the code that depends on it reads it. The GPU request
runs on a thread while Python imports pycolmap, numpy and OpenCV, so the
slowest probes overlap.

| Line | JSON key | What it is | Source |
|------|----------|------------|--------|
| `Build` | `sfmtool` | The extension's Cargo profile, `release` or `debug`, then the two `git describe` strings of the version line, either written `unknown` when it is not known. | `build_profile()`, `_build_git_describe()`, `checkout_describe` |
| `Install` | `install` | `editable` for an editable install of a checkout, `installed` otherwise, or `no package metadata`; then the directory the `sfmtool` package is run from. | `direct_url.json` of the `sfmtool` distribution ([PEP 610](https://peps.python.org/pep-0610/)): editable when its `dir_info.editable` is true |
| `Python` | `python` | The interpreter's version and path. | `platform.python_version()`, `sys.executable` |
| `OS` | `os` | The OS and its version, then the machine architecture. | `platform.platform()`, `platform.machine()` |
| `CPU` | `cpu` | The CPU's model name (`unknown model` where it cannot be read), then the logical CPUs this process may run on. | `machine::cpu_model`: the `cpuid` brand string on x86-64, the `machdep.cpu.brand_string` sysctl on Apple silicon, none elsewhere. `machine::logical_cores`: `std::thread::available_parallelism`, which counts an affinity mask or a cgroup quota |
| `Threads` | `threads` | The number of threads of the rayon pool the parallel kernels run on, followed by `(RAYON_NUM_THREADS=<value>)` when that variable is set. | `machine::rayon_threads`: `rayon::current_num_threads()` in the global pool, which honours `RAYON_NUM_THREADS` |
| `SIMD` | `simd` | The SIMD level the kernels dispatch to (below). | `machine::simd_level` |
| `Memory` | `memory_bytes` | Physical memory in GiB (2^30 bytes) to one decimal, or `unknown`. | `machine::physical_memory_bytes`: `GlobalMemoryStatusEx` on Windows, `sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE)` on Linux and macOS |
| `GPU` | `gpu` | The GPU adapter the GPU code would get (below). | `machine::probe_gpu_adapter` |
| `pycolmap` | `pycolmap` | pycolmap's version, the COLMAP version it wraps, and its CUDA support (below). | `pycolmap.__version__`, `pycolmap.COLMAP_version`, `pycolmap.has_cuda`, `pycolmap.get_num_cuda_devices` |
| `numpy` | `numpy` | numpy's version. | `numpy.__version__` |
| `OpenCV` | `opencv` | OpenCV's version, or `not installed`; it is an optional backend. | `cv2.__version__` |
| `ffmpeg` | `ffmpeg` | ffmpeg's version and path, or `not found on PATH`. `sfm insv2rig` runs `ffmpeg` from PATH to read video. | `shutil.which("ffmpeg")`, and the third word of the first line of `ffmpeg -version`, run with a five-second timeout |

A Python module that cannot be imported is reported as `not installed` when
it does not exist, and otherwise as the import's error,
`<ExceptionType>: <message>`, such as a DLL that failed to load. An `ffmpeg`
found on PATH that cannot be run, or that exits with an error, is reported
with its path and the error.

### SIMD level

The kernels in `sfmtool-core` choose their SIMD path at run time with
`is_x86_feature_detected!`, at each dispatch site, and fall back to a
narrower path or to scalar code when a feature is missing. Their
requirements come in three tiers: SSE2, which every x86-64 CPU has; AVX2
alone (the focal vote's residuals, homography estimation, the KD-forest
distances, the anisotropic remap); and AVX2 with FMA (the SIFT scale space,
orientation and descriptor, and the patch kernels). There is no AVX-512 code
and no SIMD code for another architecture. `machine::simd_level` makes the
same two checks and reports:

| Level | When |
|-------|------|
| `AVX2+FMA` | x86-64 with AVX2 and FMA: every SIMD path runs. |
| `AVX2` | x86-64 with AVX2 but not FMA: the AVX2-only kernels take their AVX2 path, and the AVX2+FMA kernels fall back. |
| `SSE2` | x86-64 without AVX2. |
| `none (scalar)` | Any other architecture, such as Apple silicon. |

The diagnostic variables `SFMTOOL_SIFT_NO_AVX2` and
`SFMTOOL_FOCAL_VOTE_NO_SIMD`, which force scalar paths in one kernel each,
are not reflected.

### GPU adapter

`machine::probe_gpu_adapter(timeout, progress)` makes the adapter request
that the GPU optical flow matcher makes in `GpuContext::new`: both call
`request_compute_adapter` in
[context.rs](../../../crates/sfmtool-core/src/features/optical_flow/gpu/context.rs),
which asks for every backend compiled in, the high-performance adapter, no
surface and no forced software fallback. The probe makes the request on a
thread of its own and waits at most `timeout`, ten seconds from
`sfm version`, so a driver that hangs cannot hang the command; the thread is
left behind and ends with the process. It never panics: a panic inside wgpu
or the driver ends that thread and is reported as a failed request. The
binding passes `Progress::none()`, so only the timeout ends the wait. The
line is one of:

| Line | JSON `gpu` | When |
|------|------------|------|
| `<name> (<backend>, <type>)` | `{"status": "found", "name", "backend", "device_type"}` | An adapter was returned. `backend` is `dx12`, `vulkan`, `metal` or `gl`; the type is `discrete`, `integrated`, `virtual`, `CPU (software)` (a software rasterizer such as lavapipe or WARP) or `other`. |
| `no GPU adapter (<reason>)` | `{"status": "no_adapter", "reason"}` | wgpu returned no adapter; the reason is its error text. |
| `no answer from the GPU driver within 10 s` | `{"status": "timed_out"}` | The request did not return in time. |
| `the GPU adapter request failed` | `{"status": "failed"}` | The request panicked. |
| `not built with GPU support` | `{"status": "not_built"}` | `sfmtool-core` was built without its `gpu` feature. |

The viewer requests its adapter differently, with its window's surface and
its platform's one backend (DX12, Metal or Vulkan), so it can run on another
adapter than the one this line names.

### pycolmap and CUDA

The line is pycolmap's version, then `COLMAP <version>` from
`pycolmap.COLMAP_version` (left out when pycolmap does not have it), then:

| Text | When |
|------|------|
| `CPU only` | `pycolmap.has_cuda` is false, as in every pixi environment but `cuda`. |
| `CUDA, <n> device(s)` | `has_cuda` is true and `pycolmap.get_num_cuda_devices()`, which only CUDA builds have, returned `n`. `CUDA, 0 devices` is a CUDA build with no usable device, such as a machine without an NVIDIA driver. |
| `CUDA build (<error>)` | `has_cuda` is true but the count failed, or this pycolmap has no `get_num_cuda_devices`. |

## JSON

`sfm version --json` prints one object with these keys, in this order. The
keys are stable, and a value that is unknown is `null`:

```json
{
  "sfmtool": {
    "version": "0.3",
    "line": "sfmtool 0.3 (v0.3-388-ge10b06192)",
    "checkout_describe": "v0.3-388-ge10b06192",
    "extension_describe": "v0.3-388-ge10b06192",
    "build_profile": "release"
  },
  "install": {"editable": true, "location": "C:\\Dev\\prod\\sfmtool\\src\\sfmtool"},
  "python": {"version": "3.14.9", "executable": "C:\\Dev\\prod\\sfmtool\\.pixi\\envs\\default\\python.exe"},
  "os": {"platform": "Windows-11-10.0.26300-SP0", "machine": "AMD64"},
  "cpu": {"model": "Intel(R) Core(TM) i9-14900HX", "logical_cores": 32},
  "threads": {"rayon": 32, "rayon_num_threads_env": null},
  "simd": "AVX2+FMA",
  "memory_bytes": 68393058304,
  "gpu": {"status": "found", "name": "NVIDIA GeForce RTX 4090 Laptop GPU",
          "backend": "vulkan", "device_type": "discrete"},
  "pycolmap": {"version": "4.2.1", "colmap_version": "4.2.1",
               "cuda_built": false, "cuda_devices": null},
  "numpy": {"version": "2.5.4"},
  "opencv": {"version": "5.0.0"},
  "ffmpeg": {"path": "C:\\Dev\\prod\\sfmtool\\.pixi\\envs\\default\\Library\\bin\\ffmpeg.EXE",
             "version": "9.0.2"}
}
```

- `sfmtool`: `version` is the package metadata version, `line` the version
  line, `checkout_describe` and `extension_describe` the two `git describe`
  strings (each `null` when unknown) and `build_profile` `"release"` or
  `"debug"`. This object alone, as `{"sfmtool": {...}}`, is what
  `--bare --json` prints.
- `install`: `editable` is `true`, `false`, or `null` without package
  metadata; `location` is the package directory.
- `python`, `os`: as in the text, each part under its own key.
- `cpu`: `model` and `logical_cores`, each `null` when unknown.
- `threads`: `rayon` is the pool's thread count and `rayon_num_threads_env`
  the value of `RAYON_NUM_THREADS`, or `null` when it is not set.
- `simd`: the level's text, as in the table above.
- `memory_bytes`: physical memory in bytes, or `null`.
- `gpu`: an object whose `status` says which other keys it has (see
  [GPU adapter](#gpu-adapter)).
- `pycolmap`: `version`, and when it imported, `colmap_version` (`null` when
  pycolmap does not give it), `cuda_built` (`has_cuda`) and `cuda_devices`,
  the device count, or `null` when pycolmap is not a CUDA build or could not
  count; `cuda_error` is added when a CUDA build could not count. When the
  import failed, `version` is `null` and `error` holds the text the line
  prints.
- `numpy`, `opencv`: `version`, or `version` `null` and `error`.
- `ffmpeg`: `path` (`null` when not on PATH) and `version`; `error` is added
  when it was found but did not run.
