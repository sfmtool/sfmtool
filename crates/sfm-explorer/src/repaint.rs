// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The callback egui calls when something asks for a repaint, which wakes the
//! event loop without calling into the window from another thread.
//!
//! egui calls it from whatever thread called `request_repaint`, with its
//! context's lock held: the GUI thread, or a rayon task that has just read a
//! photograph for a panel. On macOS `winit`'s `Window::request_redraw`, called
//! from a thread other than the main one, runs on the main thread and waits
//! until it has run there. A task that called it while the GUI thread was in a
//! frame then held egui's lock until the GUI thread went back to its event
//! loop, while the GUI thread's next read of the context waited for that lock,
//! so the viewer stopped. A call from another thread therefore sends a user
//! event through the event loop's proxy, which does not wait, and the event's
//! handler asks for the redraw on the GUI thread.

/// egui's repaint callback for a context whose frames run on the thread that
/// makes the callback: it calls `redraw` when egui calls it on that thread,
/// and `wake` when egui calls it on any other.
pub(crate) fn repaint_callback(
    redraw: impl Fn() + Send + Sync + 'static,
    wake: impl Fn() + Send + Sync + 'static,
) -> impl Fn(egui::RequestRepaintInfo) + Send + Sync + 'static {
    let gui_thread = std::thread::current().id();
    move |_info| {
        if std::thread::current().id() == gui_thread {
            redraw();
        } else {
            wake();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{mpsc, Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    /// What the worker in the test below tells the thread playing the GUI
    /// thread.
    enum Worker {
        /// Its `request_repaint` returned.
        Returned,
        /// It is in a redraw that waits for the GUI thread to run it, as
        /// `winit`'s `request_redraw` does on macOS; sending on the channel
        /// runs it.
        WaitingOnTheGuiThread(mpsc::Sender<()>),
    }

    /// A worker's repaint, asked for while the GUI thread is in a frame,
    /// never calls the redraw, which on macOS would wait on the GUI thread
    /// with egui's lock held, and wakes the loop instead; the GUI thread's
    /// next read of the context in that frame goes on. A repaint asked for on
    /// the GUI thread still redraws.
    ///
    /// The redraw is the one `winit` has on macOS: called on the GUI thread it
    /// runs at once, and called on another it hands itself to the GUI thread
    /// and waits until that thread runs it, which the GUI thread does only
    /// after the rest of its frame, a read of the context here. A callback
    /// that called it from the worker leaves both threads waiting, which the
    /// time limit turns into a failure.
    #[test]
    fn a_repaint_asked_for_on_a_worker_wakes_the_loop_rather_than_wait_on_the_gui_thread() {
        let redraws = Arc::new(AtomicUsize::new(0));
        let wakes = Arc::new(AtomicUsize::new(0));
        let (finished, frame_ended) = mpsc::channel();
        {
            let (redraws, wakes) = (Arc::clone(&redraws), Arc::clone(&wakes));
            // This thread plays the GUI thread, so the test thread can give
            // up on it.
            thread::spawn(move || {
                let gui_thread = thread::current().id();
                let (tell, told) = mpsc::channel();
                let ctx = egui::Context::default();
                ctx.set_request_repaint_callback(repaint_callback(
                    {
                        let tell = Mutex::new(tell.clone());
                        move || {
                            if thread::current().id() == gui_thread {
                                redraws.fetch_add(1, Ordering::Relaxed);
                                return;
                            }
                            let (run, ran) = mpsc::channel();
                            let _ = tell
                                .lock()
                                .unwrap()
                                .send(Worker::WaitingOnTheGuiThread(run));
                            let _ = ran.recv();
                        }
                    },
                    move || {
                        wakes.fetch_add(1, Ordering::Relaxed);
                    },
                ));
                // A repaint asked for on the GUI thread, later than the
                // worker's, so egui hands the worker's sooner one to the
                // callback too.
                ctx.request_repaint_after(Duration::from_secs(60));
                let worker = {
                    let ctx = ctx.clone();
                    thread::spawn(move || {
                        ctx.request_repaint();
                        let _ = tell.send(Worker::Returned);
                    })
                };
                let waiting = told.recv().expect("the worker tells what it did");
                // The rest of the frame: a read of the context.
                ctx.input(|_| ());
                // Back in the event loop, which runs what waits on it.
                if let Worker::WaitingOnTheGuiThread(run) = waiting {
                    let _ = run.send(());
                }
                worker.join().expect("the worker ends");
                let _ = finished.send(());
            });
        }
        frame_ended
            .recv_timeout(Duration::from_secs(10))
            .expect("the GUI thread's frame ended while the worker asked for a repaint");
        assert_eq!(
            redraws.load(Ordering::Relaxed),
            1,
            "the GUI thread's redraw"
        );
        assert_eq!(wakes.load(Ordering::Relaxed), 1, "the worker's wake");
    }
}
