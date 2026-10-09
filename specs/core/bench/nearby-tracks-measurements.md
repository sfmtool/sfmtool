# Nearby Tracks Measurements

This file records the measurements behind [nearby-tracks.md](nearby-tracks.md),
which specifies `find_nearby_tracks`: the operation that, for a pixel in one
photograph, finds the 3D points near it that other photographs see, groups them
into depth layers and builds a track for each one to put on the bench. The
measurements bear on three things the spec states: that the track-at-pixel
harness can call the operation in core in place of its own Python loop over the
pieces' bindings, that a caller that only reads the layers turns the building
of tracks off, and that the built tracks are compared so a duplicate is kept off
the bench.

All three were taken with the track-at-pixel harness
([`scripts/track_at_pixel/harness.py`](../../../scripts/track_at_pixel/harness.py))
on the two checked-in ground truths, `seoul_bull_sculpture` and `kerry_park`.
The harness removes a ground-truth point and queries each pixel the point was
observed at. It runs each query in two passes: the **full** pass removes only
the point under test, and the **empty** pass removes every point, so a query
has the cameras, the photographs, the descriptors and the clusters and no
reconstructed point. The machine was not recorded.

## Parity with the harness

**Question.** Does `find_nearby_tracks` return what the harness's own loop
(`find_anchors` with `finder_impl="python"`) returns, so that the harness can
call core by default?

**Method.** Measured with the code of PR #641. The harness was run with
`finder_impl=rust` and `finder_impl=python` over both ground truths
(seoul_bull's 1277 queries and Kerry Park's 3903), full and empty passes, with
every source and with the default stopping rule, and the two runs' anchors,
layers and summaries were compared.

**Result.** The two return the same anchors with the same sources, sightings,
classes and support, the same layers with the same members, ranks, evidence,
scores, keys and confidences, and identical summaries. The one difference is in
the last bits of a candidate's distance along its pixel's ray, which the Python
loop computes with its own camera and core with its own, and so of the range
ends found from it: at most 3e-14 relative, too small to move a class, a layer
or a reading. The two take the same time within the run-to-run noise, since the
Python loop calls the same core pieces: with the default stopping rule, about
20 ms a query in seoul_bull's full pass and 80 in Kerry Park's.

**Decision.** The harness calls core by default (`finder_impl="rust"`), and
keeps its Python loop as the reference.

## Cost of finding and building

**Question.** What does building the bench tracks cost next to finding the
candidates, and so should a caller that does not put tracks on the bench be
able to skip it?

**Method.** Measured with the code of PR #641 on seoul_bull with the default
stopping rule: once with one thread a query, and once with sixteen threads for
one query, as the viewer runs it.

**Result.** With one thread a query, finding takes about 20 ms a query in the
full pass and 5 in the empty one, and building the tracks another 57 and 75.
With sixteen threads for one query, the building takes 3 to 9 ms.

**Decision.** Building costs more than finding, so it is an option
(`tracks.build`), and a caller that only reads the layers, like the harness's
scoring, turns it off. The builds are independent and run side by side.

## How often the built tracks repeat

**Question.** How often do two candidates of one query come out as the same
track, and so would putting every built track on the bench commit duplicate
points?

**Method.** Measured with the code of PR #642 by running the harness over both
ground truths with the tracks built, once with the default stopping rule and
once with every source run, and counting the queries with a duplicate.

**Result.** With the default stopping rule, 9% of seoul_bull's queries and 8% of
Kerry Park's have one, most of them a cluster repeating another cluster; with
every source run, 62% and 34%, most of them a cluster and a guided match on the
same features. Two guided matches repeat each other when the queried image has
two keypoints at one place, which SIFT gives a feature with two orientations.
The far-field sweep's own check dropped no reading in either ground truth.

**Decision.** Duplicates are common enough that the built tracks are compared
and a duplicate is left off the bench order, as
[nearby-tracks.md](nearby-tracks.md#how-the-pieces-combine) specifies under
**Duplicates**.
