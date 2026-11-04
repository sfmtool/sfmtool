# Picking Images: Open Questions

**Status:** Draft

Amends [cli/image-feature/pick-images-command.md](../cli/image-feature/pick-images-command.md)
and [core/analysis/pick-images.md](../core/analysis/pick-images.md). Decided
with the maintainer: the command's name and its one-command shape (it reports,
and writes a list when given a name), that a written list records nothing
about how it was picked, the refusal to overwrite without `--force`, one list
per run across every sequence, the workspace and `.sift` version 2
requirement, SIFT matching through a KD-forest as the measure of motion, the
detection of a sequence of photographs, which stops the command, the target
step of 10% and the blur ratio of 0.6, and the walk.

The other defaults were run on a collection of captures: handheld walking
videos, slow orbits of an object, shaky videos with long blurred runs, a video
of a moving animal, rotation-only panoramas, videos from a tripod or a hand
held still, a numbered set of photographs of an object beside a video in one
run, and a two-sensor fisheye rig, with the lists of the solvable pinhole
videos judged by solving them. The verdicts were right on every capture; the
target step, the band, the inlier share and the panorama and barely-moves
thresholds held, and no capture came near the moving-scene threshold, whose
value is still untested. Three rules changed, and the
standing specs describe them: a small step is required of a video, a region
flagged for blur alone no longer shortens the walk's steps, and the walk takes
a step with fewer than twice `min_matches` matches only at the next frame or
when halving a failed step finds nothing better. Raising `min_matches` from 30
to 100 instead, which also mended the solve that broke at steps with few
matches, doubled the frames chosen on a shaky video, since it also flags more
of the baseline's samples for weak matches. This draft holds what is not
settled.

## Open questions

- **The output option's name.** The command uses `--write-list NAME`. `--name`
  was the maintainer's example; it reads less clearly beside the input's
  `--named-image-list`.
- **Taking `-r` and `--named-image-list` for the input.** The command is one of
  the commands of the shared `-r/--range` option, so a list or a range can
  narrow the input. Without them the input would be `PATHS` alone.
- **The default baseline size.** 200 samples read up to five frames each. On
  captures of 400 to 1,300 frames that is half to all of the frames, and more
  than the walk reads after it, which adds 30 to 260 frames; on a capture of
  6,000 frames it is a sixth. With 50 samples the same captures got the same
  verdicts, medians within a few percent and lists of about the same size,
  for 35% to 55% fewer frames read in all. But the sign that the step changes
  little compares neighbouring samples, which are then further apart, and it
  failed on two videos that it passed with 200 samples; both stayed videos
  with the other three signs. The options are to keep 200; to take one sample
  per so many frames with a floor and `--baseline-frames` as the most; a fixed
  lower default; or to read the frames 10 and 30 after only some of the
  samples. Measuring the step's change in a way that does not depend on the
  samples' spacing would go with any of the smaller ones.
- **An asymmetric band.** ±30% held on every capture, but a short step costs
  only a frame more and a long one can lose the match, so a band wider below
  the target than above it is still a choice.
- **The blur rule's spread.** A robust z of 2 against the MAD, one-sided. The
  MAD of a capture with very even blur readings is small and makes the rule
  strict; the walk floors a MAD of 0 at the smallest positive deviation
  ([core/analysis/pick-images.md § Blur](../core/analysis/pick-images.md#blur)),
  and a floor on a small but positive MAD may be needed too. A frame judged
  blurred by a z-score alone can have a sharpness above its median, which the
  report then shows as kept though blurred at more than 1 of the median. On
  the captures such a call was right on a handheld video, whose frames were
  smeared sideways at a sharpness near the median, and wrong on a walking
  video, whose sharp frames of a tilted slope have strongly directional
  gradients. Since a region flagged for blur alone no longer shortens the
  walk's steps, a wrong call costs at most a swap. The options are to keep the
  rule; to count a z-score only with a sharpness below the median; or to
  compare the tensor's direction with the direction of the motion the
  matches measure.
- **Medians per sequence or per capture.** The walk takes the sharpness and
  tensor medians per sequence, since separate sequences can differ in light and
  scene. For several sequences of one camera in one place, one median over the
  capture may be better.
- **A failing sequence among several.** The verdict stops the whole run when
  any sequence fails. The command could instead write a list of the sequences
  that pass.
- **Rigs.** Motion from the reference sensor only, sharpness per sensor against
  its own median, and a frame blurred when any sensor is. Whether a sensor
  pointing at a featureless view should be allowed to make every frame look
  blurred, and whether `.camrig` files, as well as `rig_config.json`, should
  define the rigs. A fundamental matrix does not describe two fisheye images:
  on a fisheye rig the share of matches it fits 1 frame apart was about 0.7,
  against 0.98 or more on the pinhole videos, and the walk broke 14 times and
  reported 132 weak points. Undistorting the keypoints with the rig's
  intrinsics before the fit, where `rig_config.json` gives them, would remove
  that.
- **Rig list entries.** One entry per sensor with the same range keeps each
  pattern literal but orders the list by sensor and then frame, so a slice of
  the list ([named-image-lists.md § Slices](../workspace/named-image-lists.md#slices))
  takes the first sensor's images before any other's, and the command
  suggests no slices for a rig list. One entry with a `*` segment would keep
  the sensors of a frame together but can match folders that are not sensors.
- **The exit status** when the verdict stops a run that was asked to write a
  list: the command exits 1. A flag to choose frames anyway, for a person who
  means to solve a panorama with a rotation-only tool, is not proposed.
- **The command's category.** *Image Feature*, beside `sift` and `match`, since
  it reads images and extracts features. *Workspace* is the other candidate,
  since it writes a workspace file.
- **Telling photographs from a video.** Whether a sequence with a few
  photograph-like stretches (a video paused and resumed, say) should be split
  rather than judged whole.
- **A way to choose among photographs.** How a set of photographs should be
  chosen from, for example by how much of the scene each one's matches cover
  and by sharpness alone, without steps, is a design of its own and would be
  proposed in its own draft. The command only detects such a sequence and
  stops.
- **The JSON schema** of `--json`, which is the binding's report with the
  command's keys added and has no version of its own.
- **The version 1 folder check per extraction.** `image_files_to_sift_files`
  checks, on every call that has a file to write, every `.sift` file already in
  the target folder for version 1 (`_check_versions_never_mix`), which opens
  each file. The command's source calls it once per walk step, so that check
  grows with the files the run has written, although `pick_images` has made
  the folder check once before the first extraction. `image_files_to_sift_files`
  could remember, for the life of the process, a folder it has found to hold no
  version 1 file and has since written only version 2 files into.
