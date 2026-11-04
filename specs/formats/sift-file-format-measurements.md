# Measurements behind the image readings

These measurements support the two per-image readings that a `.sift` file
records and a `.sfmr` file copies:
[the sharpness and the gradient structure tensor](sift-file-format.md#the-reading-image)
of the reading image. They bear on four decisions: that the sharpness is the
variance of the Laplacian, that the gradient structure tensor is recorded
beside it, that both are taken on one grey image reduced to a long side of
480 pixels, and that full resolution and three other blur measures were not
used. The `.sfmr` format's choice to store the sharpness as `float16` and the
tensor as `float32` rests on the ranges given in
[the `.sfmr` format's reasons](sfmr-file-format.md#why-the-readings-are-stored-this-way), not on
these measurements.

## The data and the method

Every measurement below uses two hand-held phone videos, 4K sources from one
phone type, because blur in a hand-held capture is what the readings exist to
find. Every sixth frame was graded by eye from a crop of its most textured
240-pixel block: 0 sharp, 1 soft, 2 clearly blurred (182 graded frames of one
video, 199 of the other). Unless a line says otherwise, the numbers come from a
sweep of the reading image's long side over 240 to 3840 pixels, each size area
averaged from the 4K source frames' BT.709 grey, with `z` the robust z-score of
a frame against its whole video (the median and 1.4826 times the median
absolute deviation). The scripts and the frames are not in this repository, so
the numbers cannot be repeated from it.

The evidence is two videos from one phone type, graded by one person, with
about 40 clearly blurred frames per video, so one frame is about 2.5
percentage points of recall. Synthetic blur removes sensor noise that real
blur keeps. The match-loss thresholds are interpolated between the blur levels
measured, and they are in pixels of the 1920-pixel images SIFT ran on.

## What the readings catch

The question: does each reading separate blurred frames from sharp ones within
a capture, and does the tensor add anything the Laplacian lacks?

- **The Laplacian ranks the blurred frames of a capture.** At long side 480 its
  variance separated clearly blurred from sharp frames with an AUC of 0.858 on
  one video and 0.945 on the other (Spearman ρ with the grade 0.51 and 0.65).
  Flagging frames whose `z` is above 2 caught 32% and 27% of the clearly
  blurred frames and no sharp frame. So the sharpness is the Laplacian's
  variance.
- **It depends on scene content.** A sharp frame 90% replaced by plain wall
  read `z` +8.6 and +6.4 at 480, more than a Gaussian blur of σ = 1.5 did (+1.2
  and +1.6). So the readings are compared only within a capture, and the
  format gives no threshold.
- **The tensor's axis ratio sees blur along one axis that the Laplacian
  misses.** In the second video, frames 51 to 75 are smeared horizontally and
  graded clearly blurred. The Laplacian's median `z` over them was +0.3 at 480
  and it flagged none of them. The reading that caught them is
  `|log(Exx/Eyy)|` relative to the capture's median: in an earlier measurement
  on OpenCV's reduced JPEG decode to a quarter size (long side 480, 8-bit
  BT.601 grey), the signed log ratio had a median `z` of −2.2 over the run, and
  its distance from the capture's median was beyond 2 in `z` for 64% of the
  frames, the same 64% at long side 960. The coherence and the tensor's
  distance from the capture's median normalised tensor caught 0% to 16% of that
  run at every size in the sweep. So the tensor is recorded beside the
  sharpness, raw, so that a reader can derive the axis ratio, the coherence and
  the normalised tensor from it.
- **Coherence and tensor distance add recall over all graded frames.** At 480,
  taking the larger of the Laplacian's `z` and the coherence's `z` raised the
  recall of clearly blurred frames from 32% and 27% to 35% and 44%, with no
  sharp frame flagged; with the tensor distance instead it was 42% and 44%,
  with 1% and 0% of sharp frames flagged.
- **Cost.** At 480 the resize, the Laplacian variance and the tensor took 7 ms
  on one thread from an in-memory 4K grey image in double precision, cheap
  enough to compute for every image at extraction.

## Why 480

The question: at what size of the reading image do the readings rank real blur
best, and how much does the size cost?

- **Real blur ranks best at 360 to 640.** The Laplacian's AUC on the graded
  frames (first video / second video) was 0.848/0.948 at 360, 0.858/0.945 at
  480 and 0.851/0.939 at 640, and fell from 960 up: 0.841/0.920 at 960,
  0.838/0.879 at 1920 and 0.832/0.857 on the 4K source.
- **Synthetic blur, and the blur that matters for matching, rank better at
  larger sizes.** For synthetic copies blurred just enough to cost about 20% of
  a frame's verified matches ([below](#what-blur-costs-matching)), the AUC
  against untouched frames was 0.74/0.81 at 240, 0.90/0.89 at 480, 0.94/0.92 at
  640, 0.98/0.94 at 960 and 1.00/0.96 at 1920. The likely reason, not tested,
  is that synthetic blur also removes sensor noise and compression texture,
  which real blur keeps, so the larger sizes that see that texture separate
  synthetic copies more easily than real blurred frames. The real frames
  decide the size.
- **Small sizes are pulled more by plain areas.** The sharp frame 90% plain
  wall read `z` +8.6 at 480 against +5.8 at 960 in the first video, where a
  Gaussian blur of σ 1.5 read +1.2 and +2.8.
- **Frame-to-frame jitter grows with size**: the median change between
  consecutive sharp frames, in units of the capture's spread, was 0.10/0.11 at
  480 and 0.16/0.13 at 1920.
- **The choice within the plateau.** 480 is the middle of 360 to 640; it divides
  3840 and 1920 by whole numbers, so area averaging is exact there; at 7 ms it
  is cheap; and it jitters less than 640 in the first video (0.096 against
  0.108) and about as much in the second (0.114 against 0.113). So the reading
  image's long side is 480.
- **One reading image for both values.** The coherence is slightly better at
  960: with the Laplacian at 480, the combined recall was 42% and 49% with the
  coherence at 960 against 35% and 44% at 480. That is two or three frames of
  about 40 clearly blurred frames per video, which does not justify a second
  image, so both readings are taken on the one 480-pixel image.

## What blur costs matching

The question: how much blur costs a frame its matches, so that the sizes above
can be judged against the blur that matters for a solve?

One frame of each of 40 sharp pairs (20 per video) was blurred at the 4K source
and resized to 1920 pixels, and the sfmtool SIFT matches that survive geometric
verification against the unblurred neighbour were counted against the same
pair unblurred. In pixels of the 1920-pixel images SIFT ran on, the pooled
median loss reached 20% at a Gaussian σ of 1.43 px and 50% at 2.39 px, and 20%
at a linear motion of 5.5 px and 50% at 9.4 px. Blur of σ ≤ 1 cost a pooled
median of at most about 6% (16% in the first video at σ = 1). So the mild blur
that a 480-pixel reading image misses mostly does not matter for the solve.

## Measured and not chosen

The question: does another size or another blur measure do better than the
Laplacian at 480?

- **Full resolution**: the Laplacian on the 4K source ranked worse (AUC 0.832
  and 0.857 in the sweep) and costs about ten times as much (78 ms against
  7 ms).
- **Tenengrad** (the mean of the Sobel gradient energy): AUC 0.795 and 0.887 on
  the earlier quarter decode, and more sensitive to plain content than the
  Laplacian.
- **Sum-modified Laplacian**: AUC 0.852 and 0.932 on the earlier quarter
  decode, no better than the Laplacian and more sensitive to plain content.
- **ZNCC self-similarity on 24-pixel crops**: AUC 0.899 and 0.939 on the
  earlier quarter decode, but its best capture-relative rule flagged no blurred
  frame in the first video, and it took 14.9 ms a frame.

None of them is recorded.
