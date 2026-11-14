// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The two sources on real images and missing files, and the worker pool's
//! order, replacement, cancellation and panics.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sfmtool_sift_format::read_sift_metadata;

use super::*;
use crate::features::sift::build_sift_files;

const SEOUL_BULL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test-data/images/seoul_bull_sculpture"
);

/// Copies of the first `count` seoul_bull images in a temporary folder.
fn images(count: usize) -> (tempfile::TempDir, Vec<PathBuf>) {
    let dir = tempfile::tempdir().unwrap();
    let paths = (1..=count)
        .map(|i| {
            let name = format!("seoul_bull_sculpture_{i:02}.jpg");
            let to = dir.path().join(&name);
            std::fs::copy(Path::new(SEOUL_BULL).join(&name), &to).unwrap();
            to
        })
        .collect();
    (dir, paths)
}

fn id(frame: usize) -> FrameId {
    FrameId { sequence: 0, frame }
}

/// One single-image frame per image, its `.sift` file where `options` puts
/// it.
fn frames(images: &[PathBuf], options: &BuildSiftFilesOptions) -> Vec<CaptureFrame> {
    images
        .iter()
        .enumerate()
        .map(|(t, image)| CaptureFrame {
            number: t as i64 + 1,
            images: vec![CaptureImage {
                image_path: image.clone(),
                sift_path: options.sift_path_for_image(image),
            }],
        })
        .collect()
}

fn requests<'a>(frames: &'a [CaptureFrame], which: &[usize]) -> Vec<(FrameId, &'a CaptureFrame)> {
    which.iter().map(|&t| (id(t), &frames[t])).collect()
}

fn content_hash(path: &Path) -> String {
    let (_, _, hash) = read_sift_metadata(path).unwrap();
    format!("{hash:?}")
}

#[test]
fn the_extracting_source_writes_what_build_sift_files_writes() {
    let (dir, paths) = images(3);
    let options = BuildSiftFilesOptions {
        concurrent_images: Some(2),
        ..BuildSiftFilesOptions::sfmtool_defaults()
    };
    let frames = frames(&paths, &options);
    let mut source = ExtractingSource::new(options.clone()).unwrap();
    assert_eq!(source.concurrency(), 2);
    source
        .wait(&requests(&frames, &[0, 1, 2]), &Progress::none())
        .unwrap();
    assert_eq!(source.finish().written, vec![id(0), id(1), id(2)]);

    let (_other, copies) = images(3);
    let built = build_sift_files(&copies, &options, &Progress::none()).unwrap();
    for (frame, sift) in frames.iter().zip(&built.sift_paths) {
        assert_eq!(content_hash(&frame.images[0].sift_path), content_hash(sift));
    }
    // Up to date now, so a second source writes nothing.
    let mut again = ExtractingSource::new(options).unwrap();
    again
        .wait(&requests(&frames, &[2, 0]), &Progress::none())
        .unwrap();
    assert!(again.finish().written.is_empty());
    drop(dir);
}

#[test]
fn a_prefetched_frame_is_written_without_a_wait() {
    let (_dir, paths) = images(2);
    let options = BuildSiftFilesOptions {
        concurrent_images: Some(1),
        ..BuildSiftFilesOptions::sfmtool_defaults()
    };
    let frames = frames(&paths, &options);
    let mut source = ExtractingSource::new(options).unwrap();
    source.prefetch(&requests(&frames, &[1]));
    let sift = &frames[1].images[0].sift_path;
    let start = Instant::now();
    while !sift.exists() && start.elapsed() < Duration::from_secs(60) {
        std::thread::sleep(Duration::from_millis(10));
    }
    // The frame waited for after it is written too, and the first is not.
    source
        .wait(&requests(&frames, &[1]), &Progress::none())
        .unwrap();
    assert_eq!(source.finish().written, vec![id(1)]);
    assert!(sift.exists());
    assert!(!frames[0].images[0].sift_path.exists());
}

#[test]
fn an_image_that_cannot_be_read_is_an_error_naming_the_frame() {
    let (dir, mut paths) = images(1);
    let broken = dir.path().join("broken_0002.jpg");
    std::fs::write(&broken, b"not a jpeg").unwrap();
    paths.push(broken.clone());
    paths.push(dir.path().join("missing_0003.jpg"));
    let options = BuildSiftFilesOptions::sfmtool_defaults();
    let frames = frames(&paths, &options);
    let mut source = ExtractingSource::new(options).unwrap();
    // The first failing frame in the order waited for is the error.
    let error = source
        .wait(&requests(&frames, &[0, 2, 1]), &Progress::none())
        .unwrap_err();
    match &error {
        FrameSourceError::Extract {
            frame,
            error: BuildSiftFilesError::Image { path, .. },
        } => {
            assert_eq!(*frame, id(2));
            assert_eq!(path, &paths[2]);
        }
        other => panic!("{other:?}"),
    }
    match source.wait(&requests(&frames, &[1]), &Progress::none()) {
        Err(FrameSourceError::Extract {
            frame,
            error: BuildSiftFilesError::Image { path, .. },
        }) => {
            assert_eq!(frame, id(1));
            assert_eq!(path, broken);
        }
        other => panic!("{other:?}"),
    }
    assert!(error.to_string().contains("missing_0003.jpg"), "{error}");
    assert_eq!(source.finish().written, vec![id(0)]);
}

#[test]
fn a_sift_path_elsewhere_is_refused() {
    let (dir, paths) = images(1);
    let options = BuildSiftFilesOptions::sfmtool_defaults();
    let mut frames = frames(&paths, &options);
    let elsewhere = dir.path().join("elsewhere.sift");
    frames[0].images[0].sift_path = elsewhere.clone();
    let mut source = ExtractingSource::new(options.clone()).unwrap();
    match source.wait(&requests(&frames, &[0]), &Progress::none()) {
        Err(FrameSourceError::SiftPath {
            frame,
            sift_path,
            expected,
        }) => {
            assert_eq!(frame, id(0));
            assert_eq!(sift_path, elsewhere);
            assert_eq!(expected, options.sift_path_for_image(&paths[0]));
        }
        other => panic!("{other:?}"),
    }
    assert!(!elsewhere.exists());
}

#[test]
fn invalid_options_and_version_1_settings_are_refused() {
    let mut invalid = BuildSiftFilesOptions::sfmtool_defaults();
    invalid.feature_options["octave_layers"] = serde_json::json!(0);
    assert!(matches!(
        ExtractingSource::new(invalid),
        Err(BuildSiftFilesError::FeatureOptions(_))
    ));
    let mut version_1 = BuildSiftFilesOptions::sfmtool_defaults();
    version_1
        .feature_options
        .as_object_mut()
        .unwrap()
        .remove("image_to_gray");
    assert!(matches!(
        ExtractingSource::new(version_1),
        Err(BuildSiftFilesError::Version1Settings)
    ));
}

#[test]
fn the_read_only_source_names_a_missing_file() {
    let dir = tempfile::tempdir().unwrap();
    let present = dir.path().join("a.jpg.sift");
    std::fs::write(&present, b"anything").unwrap();
    let missing = dir.path().join("b.jpg.sift");
    let frame = |sift: &Path| CaptureFrame {
        number: 1,
        images: vec![CaptureImage {
            image_path: dir.path().join("x.jpg"),
            sift_path: sift.to_path_buf(),
        }],
    };
    let frames = [frame(&present), frame(&missing)];
    let mut source = ReadOnlySource;
    assert_eq!(source.concurrency(), 0);
    source.prefetch(&requests(&frames, &[1]));
    source
        .wait(&requests(&frames, &[0]), &Progress::none())
        .unwrap();
    let error = source
        .wait(&requests(&frames, &[0, 1]), &Progress::none())
        .unwrap_err();
    assert_eq!(
        error,
        FrameSourceError::Missing {
            frame: id(1),
            path: missing.clone()
        }
    );
    assert!(error.to_string().contains("b.jpg.sift"));
    assert!(source.finish().written.is_empty());
    assert!(!missing.exists());
}

/// A frame with one image, whose paths the pool's test jobs do not use.
fn frame(t: usize) -> CaptureFrame {
    CaptureFrame {
        number: t as i64,
        images: vec![CaptureImage {
            image_path: PathBuf::from(format!("{t}.jpg")),
            sift_path: PathBuf::from(format!("{t}.jpg.sift")),
        }],
    }
}

/// A pool of `threads` workers whose job records each frame it runs and
/// waits until `gate` is open.
fn gated_pool(threads: usize, gate: Arc<AtomicBool>) -> (WorkerPool, Arc<Mutex<Vec<FrameId>>>) {
    let ran = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&ran);
    let job: Arc<Job> = Arc::new(move |id, _images, progress, wrote| {
        log.lock().unwrap().push(id);
        while !gate.load(Ordering::Relaxed) {
            if progress.is_cancelled() {
                return Err(FrameSourceError::Cancelled);
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        *wrote = true;
        Ok(())
    });
    (WorkerPool::new(job, Threads::Fixed(threads)), ran)
}

#[test]
fn a_wait_goes_ahead_of_prefetched_frames_and_a_prefetch_replaces_the_last() {
    let frames: Vec<CaptureFrame> = (0..12).map(frame).collect();
    let gate = Arc::new(AtomicBool::new(false));
    let (mut pool, ran) = gated_pool(1, Arc::clone(&gate));
    pool.prefetch(&requests(&frames, &[0, 1, 2, 3, 4, 5]));
    // The one worker holds frame 0 until the gate opens.
    let start = Instant::now();
    while ran.lock().unwrap().is_empty() && start.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(1));
    }
    // A new prediction drops frames 1 to 5 and keeps 3; the wait on 9 goes
    // first.
    pool.prefetch(&requests(&frames, &[3, 10]));
    std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(50));
            gate.store(true, Ordering::Relaxed);
        });
        pool.wait(&requests(&frames, &[9]), &Progress::none())
    })
    .unwrap();
    // The worker goes on to the prefetched frames in their order; a wait
    // made before it started them would put them in the wait's order.
    let start = Instant::now();
    while ran.lock().unwrap().len() < 4 && start.elapsed() < Duration::from_secs(10) {
        std::thread::sleep(Duration::from_millis(1));
    }
    pool.wait(&requests(&frames, &[10, 3]), &Progress::none())
        .unwrap();
    assert_eq!(*ran.lock().unwrap(), vec![id(0), id(9), id(3), id(10)]);
    assert_eq!(pool.finish().written, vec![id(0), id(3), id(9), id(10)]);
}

#[test]
fn a_cancelled_wait_stops_the_workers_promptly() {
    let frames: Vec<CaptureFrame> = (0..8).map(frame).collect();
    // A gate that never opens: every job runs until it is stopped.
    let (mut pool, ran) = gated_pool(3, Arc::new(AtomicBool::new(false)));
    pool.prefetch(&requests(&frames, &[1, 2, 3, 4]));
    let flag = Arc::new(AtomicBool::new(false));
    let start = Instant::now();
    // The wait runs on a thread of its own, so that a wait that ignores the
    // cancellation fails this test at the timeout instead of hanging it.
    let (sender, receiver) = std::sync::mpsc::channel();
    let wait_flag = Arc::clone(&flag);
    std::thread::spawn(move || {
        let progress = Progress::none().cancelled_by(&wait_flag);
        let result = pool.wait(&requests(&frames, &[0]), &progress);
        let _ = sender.send((result, pool));
    });
    std::thread::sleep(Duration::from_millis(100));
    flag.store(true, Ordering::Relaxed);
    let (result, mut pool) = receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("the cancelled wait did not return within 5 s");
    assert_eq!(result, Err(FrameSourceError::Cancelled));
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "{:?}",
        start.elapsed()
    );
    // Three frames started and were stopped; none was written.
    assert_eq!(ran.lock().unwrap().len(), 3);
    assert!(pool.finish().written.is_empty());
}

#[test]
fn a_panic_in_a_job_is_an_error_not_a_hang() {
    let frames: Vec<CaptureFrame> = (0..2).map(frame).collect();
    let job: Arc<Job> = Arc::new(|id, _images, _progress, _wrote| {
        if id.frame == 1 {
            panic!("a bug in the extractor");
        }
        Ok(())
    });
    let mut pool = WorkerPool::new(job, Threads::Fixed(2));
    match pool.wait(&requests(&frames, &[0, 1]), &Progress::none()) {
        Err(FrameSourceError::Panicked { frame, message }) => {
            assert_eq!(frame, id(1));
            assert!(message.contains("a bug in the extractor"), "{message}");
        }
        other => panic!("{other:?}"),
    }
    assert!(pool.finish().written.is_empty());
}

#[test]
fn a_rig_frame_stopped_between_its_images_reports_the_file_written() {
    // One frame of two images; the progress cancels once the first is
    // written, so the second stops at its first check.
    let (_dir, paths) = images(2);
    let options = BuildSiftFilesOptions::sfmtool_defaults();
    let frame = CaptureFrame {
        number: 1,
        images: paths
            .iter()
            .map(|image| CaptureImage {
                image_path: image.clone(),
                sift_path: options.sift_path_for_image(image),
            })
            .collect(),
    };
    let params = SiftParams::from_feature_options(&options.feature_options).unwrap();
    let flag = AtomicBool::new(false);
    let sink = |event: crate::progress::Event<'_>| {
        if let crate::progress::Event::Count { done: 1, .. } = event {
            flag.store(true, Ordering::Relaxed);
        }
    };
    let progress = Progress::to(&sink).cancelled_by(&flag);
    let mut wrote = false;
    let result = extract_frame(
        id(0),
        &frame.images,
        &options,
        &params,
        &progress,
        &mut wrote,
    );
    assert_eq!(result, Err(FrameSourceError::Cancelled));
    assert!(wrote);
    assert!(frame.images[0].sift_path.exists());
    assert!(!frame.images[1].sift_path.exists());
}

#[test]
fn a_frame_stopped_after_writing_is_counted_as_written() {
    // A job that writes and is then stopped, as a rig frame stopped between
    // its images is; and one that writes and then fails.
    let frames: Vec<CaptureFrame> = (0..2).map(frame).collect();
    let ran = Arc::new(AtomicBool::new(false));
    let first_ran = Arc::clone(&ran);
    let job: Arc<Job> = Arc::new(move |id, images, _progress, wrote| {
        *wrote = true;
        if id.frame == 0 {
            first_ran.store(true, Ordering::Relaxed);
            Err(FrameSourceError::Cancelled)
        } else {
            Err(FrameSourceError::Missing {
                frame: id,
                path: images[0].sift_path.clone(),
            })
        }
    });
    let mut pool = WorkerPool::new(job, Threads::Fixed(1));
    pool.prefetch(&requests(&frames, &[0]));
    let start = Instant::now();
    while !ran.load(Ordering::Relaxed) && start.elapsed() < Duration::from_secs(10) {
        std::thread::yield_now();
    }
    assert!(pool
        .wait(&requests(&frames, &[1]), &Progress::none())
        .is_err());
    assert_eq!(pool.finish().written, vec![id(0), id(1)]);
}

#[test]
fn workers_stopped_as_they_start_all_stop() {
    // Workers asked to stop just as they start, before or after they first
    // find no work, all see the stop, so `finish` joins every one. A worker
    // that missed the stop's wake-up would sleep forever, and the watchdog
    // reports the hang.
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let frames: Vec<CaptureFrame> = (0..1).map(frame).collect();
        for round in 0..500 {
            let job: Arc<Job> = Arc::new(|_id, _images, _progress, _wrote| Ok(()));
            let mut pool = WorkerPool::new(job, Threads::Fixed(8));
            pool.prefetch(&requests(&frames, &[0]));
            pool.finish();
            let _ = done.send(round);
        }
    });
    let mut rounds = 0;
    while rounds < 500 {
        match finished.recv_timeout(Duration::from_secs(20)) {
            Ok(_) => rounds += 1,
            Err(_) => panic!("finish hung in round {rounds}: a worker missed the stop"),
        }
    }
}
