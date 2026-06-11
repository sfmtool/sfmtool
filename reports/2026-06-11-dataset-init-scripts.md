# Dataset reconstruction scripts — parameters and results

_Updated 2026-06-11._

The four `scripts/init_dataset_*.sh` workspaces each generate an `sfm_solve.sh`
that runs SIFT extraction, matching, and SfM. They use the sfmtool SIFT backend
and the track-cluster matcher (`sfm match --cluster`). This note records each
script's configuration and its end-to-end results against the previous versions.

## Configuration

| script | images | SIFT (feature cap) | matcher | solver |
|---|--:|---|---|---|
| seoul_bull | 17 | sfmtool | cluster, `d=28` | incremental |
| seattle_backyard | 26 | sfmtool, max 2000 | cluster | global |
| kerry_park | 48 | sfmtool | cluster | global |
| dino_dog_toy | 85 | sfmtool, max 2500 | cluster | incremental |

## Results

End-to-end wall-clock of `sfm_solve.sh` (ws init + SIFT + match + solve) from a
fresh workspace, minimum of repeated runs. Registered images and 3D points are
read from the resulting `.sfmr`.

| script | old time | old reg | old points | new time | new reg | new points |
|---|--:|:-:|--:|--:|:-:|--:|
| seoul_bull | 14.3 s | 17/17 | 1,080 | 6.7 s | 17/17 | 1,061 |
| seattle_backyard | 8.4 s | 26/26 | 507 | 8.9 s | 26/26 | 3,337 |
| kerry_park | 48.2 s | 48/48 | 1,182 | 13.6 s | 48/48 | 753 |
| dino_dog_toy | 230.0 s | 85/85 | 5,281 | 81.5 s | 85/85 | 19,009 |

## Parameter rationale

- **SIFT backend (all scripts):** the sfmtool extractor is used; `sfm ws init
  --max-features` configures its feature cap.
- **Matcher (all scripts):** `sfm match --cluster` at its defaults — `accurate`
  forest preset, background rank `d=10`, radius `α=0.8` — except where noted.
- **seoul_bull — `d=28`, incremental:** the 17 images are 270×480 with sparse
  features. At the default floor `d=10` the incremental solver registers 8 of 17;
  widening the floor to `d=28` registers all 17. Global SfM produced a mirrored
  (chirality-flipped) reconstruction on these matches, so incremental is used.
- **dino_dog_toy — max 2500, incremental:** the high-resolution images reach the
  extractor's default cap of 8192 features each. Capping at 2500 reduces match and
  solve time while retaining ~19k points.
- **seattle_backyard — max 2000, global:** features are capped at 2000; global SfM
  registers all 26 images.
- **kerry_park — global:** the fisheye rig uses global SfM, features uncapped.
