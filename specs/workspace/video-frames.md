# `video-frames.json`: Where Extracted Frames Came From

A directory of frames extracted from one video holds, beside the frames, a
small JSON file named `video-frames.json` that records which video the frames
came from, how they were extracted, how an HDR video's pixels were
converted, and the time in the video of every frame.
The frame files are numbered by the order they were written, not by their
place in the video, and a phone's video has a variable frame rate, so a
frame's number gives its time by no formula; once the video is gone or edited,
this file is the only record of the times. It lets a later step match frames
to another record of the same capture (a GPS track, a second camera) or
extract one stretch of the video again at a higher rate.

## Location and encoding

The file is named `video-frames.json` and sits in the directory that holds
the frames it describes, which holds the frames of exactly one video. It is a
UTF-8 JSON object. Its `.json` extension keeps it out of every listing of
image files.

## Example

```json
{
  "version": 2,
  "video": "/captures/raw/walk1.mp4",
  "video_stream": 0,
  "codec": "hevc",
  "video_size": [3840, 2160],
  "rotation_degrees": 90,
  "color_transfer": "arib-std-b67",
  "hdr_conversion": {"transfer": "HLG", "recipe": "sfmtool-hdr-to-srgb", "recipe_version": 1, "exposure_stops": 0},
  "fps": 10,
  "max_frame_size": 1920,
  "frame_size": [1080, 1920],
  "frame_pattern": "frame_%06d.jpg",
  "frame_times_s": [0.0, 0.1, 0.2, 0.300222, 0.4]
}
```

## Keys

Every key is present in a version 2 file. Version 1 had no
`hdr_conversion`, since its writer refused HDR video; a reader of version 2
reads a version 1 file as having `hdr_conversion` `null`.

| Key | Value |
|-----|-------|
| `version` | `2`, the version of this layout. A reader refuses a version it does not know. |
| `video` | The absolute path of the video file, as it was when the frames were extracted. |
| `video_stream` | Which of the file's video streams the frames come from, counted from 0 in file order among the streams whose type is video, including a still picture stored as one, such as cover art. |
| `codec` | The name of the stream's compression format, in FFmpeg's naming (`h264`, `hevc`, `mpeg4`). |
| `video_size` | `[width, height]` of the stream's pixels as stored, before the display rotation and before any scaling to square pixels. |
| `rotation_degrees` | The rotation the video's metadata asks a player to apply when it displays the stream, in degrees, positive counterclockwise; `0` when the metadata asks for none. The frames are written already turned by it. |
| `color_transfer` | The stream's transfer characteristic, in FFmpeg's naming (`bt709`, `smpte170m`, `arib-std-b67`, `smpte2084`), or `null` when the video does not state one. |
| `hdr_conversion` | `null` when the frames hold the video's own 8-bit rendering, an SDR video's. For an HDR video, whose frames were converted to sRGB by a fixed recipe, an object: `transfer`, the HDR transfer function the video is coded with, `"HLG"` (hybrid log-gamma, `color_transfer` `arib-std-b67`) or `"PQ"` (perceptual quantizer, `smpte2084`); `recipe`, the name of the conversion, `"sfmtool-hdr-to-srgb"`; `recipe_version`, its version, an integer, a version giving each pixel value one 8-bit result; `exposure_stops`, the exposure change applied to every frame, display light having been multiplied by `2^exposure_stops` before the highlight roll-off, `0` for the recipe's own anchor. Every frame of the directory was converted with the same values. Version 1 of the recipe maps BT.2100 HLG or PQ display light to BT.709 primaries and the sRGB encoding, 203 cd/m² being sRGB's 1.0 and light from 87.84 to 1000 cd/m² compressed into the range up to 203 cd/m², so that HDR reference white (203 cd/m²) becomes sRGB code 229. |
| `fps` | The frame rate the frames were sampled at: a positive number of frames per second of video time, at most one frame being kept from each interval `[k/fps, (k+1)/fps)` of presentation time, the earliest frame in it; or the string `"full"`, every frame of the stream. |
| `max_frame_size` | The largest long side, in pixels, a frame was allowed: an integer, frames whose long side was larger having been scaled down keeping their aspect, smaller ones written at their own size; or the string `"full"`, every frame at the stream's size. |
| `frame_size` | `[width, height]` of the frame files, in pixels, upright, with square pixels. Every frame has this size. |
| `frame_pattern` | The frame files' names in the directory, as a printf pattern with one integer field: `frame_%06d.jpg` names `frame_000001.jpg`, `frame_000002.jpg`, … The numbers run from 1 with no gaps, in the order of the frames' presentation times. A number wider than the field is written in full (`frame_1000000.jpg`). |
| `frame_times_s` | One number per frame: element `i` is the presentation time of frame number `i + 1`, in seconds from the start of the video file, rounded to the microsecond. The times increase strictly. Its length is the number of frames. |

## Rules for a consumer

- The frame with number `n` is the file `frame_pattern` names with `n`, and
  its time is `frame_times_s[n - 1]`.
- A frame file can have been deleted after extraction, for example after
  choosing which frames to keep. The record still lists every frame extracted,
  so the numbers of the frames that remain keep their times.
- A program that adds frames to the directory, or renumbers them, writes the
  record again to match, or deletes it.

## Implementations

`sfm extract-frames` writes the file
([cli/workspace/extract-frames-command.md](../cli/workspace/extract-frames-command.md)),
and so does `sfm pick-images --new-workspace` with the same code
([cli/image-feature/pick-images-command.md](../cli/image-feature/pick-images-command.md#from-videos---new-workspace)),
in [`src/sfmtool/_video_frames.py`](../../src/sfmtool/_video_frames.py). Its
`--prune-images` deletes frames and leaves the record as it is. No sfmtool
command reads it. The HDR conversion version 1 is specified in
[core/camera/hdr-to-srgb.md](../core/camera/hdr-to-srgb.md).
