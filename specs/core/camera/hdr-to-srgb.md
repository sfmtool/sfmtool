# Converting HDR Video Frames to sRGB

Phones record HDR video: 10-bit pixels whose values are not the brightness
an ordinary (SDR) image stores, but a signal coded with an HDR transfer
function, hybrid log-gamma (HLG) or the perceptual quantizer (PQ), over the
wide BT.2020 colour gamut. Written to 8-bit JPEG as if it were SDR, such a
frame comes out flat and desaturated (HLG) or dark and flat (PQ). sfmtool
converts it with one fixed recipe: every pixel of every frame goes through
the same function of its own value, with no analysis of the frame, the scene
or the video (no per-frame peak detection, no local tone mapping, no dynamic
metadata). Frames from different times and different videos of a capture so
get the same exposure, which a feature matcher and a photometric step rely
on. The recipe follows ITU-R BT.2100 for the signal, ITU-R BT.2408 for the
anchor (HDR reference white, 203 cd/m², is the SDR scale's white) and the
BT.2390 EETF for the highlight roll-off.

## Interface

```rust
pub const HDR_TO_SRGB_RECIPE: &str = "sfmtool-hdr-to-srgb";
pub const HDR_TO_SRGB_RECIPE_VERSION: u32 = 1;
pub const HDR_EXPOSURE_STOPS_LIMIT: f32 = 16.0;

pub enum HdrTransfer { Hlg, Pq }
pub struct HdrToSrgbOptions { pub transfer: HdrTransfer, pub exposure_stops: f32 }
pub enum HdrToSrgbError { SampleCount { width: u32, height: u32, samples: usize }, Exposure(f32) }

pub fn hdr_ycbcr_to_srgb(
    planes: &[u16], width: u32, height: u32, options: &HdrToSrgbOptions,
) -> Result<ImageU8, HdrToSrgbError>;
```

```python
from sfmtool.fileio import write_image_from_hdr

# planes: (3, H, W) uint16, Y'CbCr at 16 bits, as ffmpeg's yuv444p16le
write_image_from_hdr("frame_000001.jpg", planes, transfer="HLG")
write_image_from_hdr("frame_000001.jpg", planes, transfer="PQ", exposure_stops=-0.5)
```

The Rust function is in
[camera/image/hdr_to_srgb.rs](../../../crates/sfmtool-core/src/camera/image/hdr_to_srgb.rs)
and the binding in
[fileio/image.rs](../../../crates/sfmtool-py/src/fileio/image.rs).

The input is three planes, Y' then Cb then Cr, of `width * height` 16-bit
codes each: FFmpeg's `yuv444p16le`, the form a decoder hands over without
converting to RGB, so the matrix and the range are applied here, where they
are stated, rather than by a scaler whose arithmetic varies between builds.
The output is a 3-channel `ImageU8` in sRGB. The rows are converted in
parallel with rayon; each pixel is converted alone, so the output does not
depend on the thread count, and identical pixels give identical output
wherever they are.

The binding takes the planes as a `(3, H, W)` `uint16` array and a
`transfer` of `"HLG"` or `"PQ"`, converts the frame and writes it with
`ImageU8::write`, as `write_image_rgb` writes an image (a JPEG at quality 95,
4:2:0), all with the GIL released. It converts and writes in one call because
its only caller writes every converted frame to a file, so the sRGB array is
never handed to Python. It also exports the three constants above.

Errors: `SampleCount` when `planes` does not hold `3 * width * height`
codes, and `Exposure` for an `exposure_stops` that is not finite or is beyond
`HDR_EXPOSURE_STOPS_LIMIT` either way, a limit that keeps every scaled light
far inside `f32`'s range. The binding raises `TypeError` for an array that is
not `uint16` and `ValueError` for a shape other than `(3, H, W)` with `H` and
`W` at least 1, an unknown transfer, a bad exposure or quality, and
`write_image_rgb`'s other errors.

`HDR_TO_SRGB_RECIPE` and `HDR_TO_SRGB_RECIPE_VERSION` name the recipe in the
record written beside the frames
([video-frames.md](../../workspace/video-frames.md)). Any change that alters
an output code, a constant or a step below, takes a new version.

## The recipe, version 1

Each pixel's three codes `D_Y, D_Cb, D_Cr` go through these steps, in `f32`.

1. **Y'CbCr to R'G'B'**, BT.2100 narrow range and BT.2020 non-constant
   luminance (`K_R = 0.2627`, `K_B = 0.0593`, `K_G = 0.6780`):
   `Y' = (D_Y/256 − 16)/219`, `Cb = (D_Cb/256 − 128)/224`,
   `Cr = (D_Cr/256 − 128)/224`;
   `R' = Y' + 1.4746 Cr`, `G' = Y' − 0.164553 Cb − 0.571353 Cr`,
   `B' = Y' + 1.8814 Cb` (the G' coefficients are `2 K_B (1 − K_B)/K_G` and
   `2 K_R (1 − K_R)/K_G`). Each is clamped to `[0, 1]`: footroom below black
   is black and headroom above nominal peak is the peak.
2. **Display light**, BT.2020 RGB in cd/m²:
   - **HLG.** The inverse OETF per channel gives scene light,
     `E = E'²/3` for `E' ≤ 1/2` and `E = (exp((E' − c)/a) + b)/12` above,
     with `a = 0.17883277`, `b = 1 − 4a = 0.28466892`,
     `c = 0.5 − a ln(4a) = 0.55991073`. The OOTF for a display of peak
     `L_W = 1000` cd/m² and black 0 then gives display light:
     `Y_s = 0.2627 R + 0.6780 G + 0.0593 B`,
     `F = L_W · Y_s^(γ − 1) · E` per channel, with the system gamma
     `γ = 1.2 + 0.42 log10(L_W/1000) = 1.2` (and `F = 0` where `Y_s = 0`).
     HLG 75%, BT.2408's reference white, gives 203.15 cd/m².
   - **PQ.** The EOTF per channel,
     `F = 10000 · (max(E'^(1/m2) − c1, 0)/(c2 − c3 E'^(1/m2)))^(1/m1)`,
     with `m1 = 2610/16384`, `m2 = 2523/32`, `c1 = 3424/4096`,
     `c2 = 2413/128`, `c3 = 2392/128`.
3. **Exposure.** `F · 2^s`, `s` the exposure in stops. The scale is applied
   to display light, before the roll-off, so a negative `s` brings highlights
   out of the roll-off and a positive one pushes more of the frame into it.
4. **BT.2020 to BT.709 primaries**, linear RGB, D65 white to D65 white, by
   the matrix derived from the two sets of chromaticities (ITU-R BT.2087):
   ```text
   [ 1.6604910021  -0.5876411388  -0.0728498633 ]
   [-0.1245504745   1.1328998971  -0.0083494226 ]
   [-0.0181507634  -0.1005788980   1.1187296614 ]
   ```
   Each row sums to 1, and the product is evaluated as
   `F_i + Σ_{j≠i} M_ij (F_j − F_i)`, so a grey stays exactly the same grey.
5. **Out-of-gamut colours.** A component below 0 after step 4 (a colour
   BT.2020 holds and BT.709 does not, such as a saturated green) is set to 0,
   per channel. This is the simplest deterministic rule; it moves such a
   colour's hue a little toward the nearest BT.709 primary or secondary.
6. **Highlight roll-off**, the BT.2390 EETF applied to `m = max(R, G, B)`,
   with source white `L_W = 1000` cd/m², target white `L_T = 203` cd/m² and
   both blacks 0, in the PQ domain normalised by the source white. With
   `PQ⁻¹(L) = ((c1 + c2 y^m1)/(1 + c3 y^m1))^m2`, `y = L/10000`:
   `maxLum = PQ⁻¹(203)/PQ⁻¹(1000) = 0.772370`,
   `KS = 1.5 · maxLum − 0.5 = 0.658555` (the knee, at 87.84 cd/m²).
   For `m` below the knee the pixel is unchanged. Otherwise
   `E1 = min(PQ⁻¹(m)/PQ⁻¹(1000), 1)`, `T = (E1 − KS)/(1 − KS)`,
   `E2 = (2T³ − 3T² + 1) KS + (T³ − 2T² + T)(1 − KS) + (−2T³ + 3T²) maxLum`,
   `m' = PQ(E2 · PQ⁻¹(1000))`, and R, G and B are each multiplied by `m'/m`.
   One factor for the three channels keeps the hue and the ratios of the
   channels; the largest channel of 1000 cd/m² or more reaches 203 cd/m². The
   EETF's black-level term vanishes with both blacks at 0.
7. **Anchor.** `y = RGB/203`: HDR reference white, 203 cd/m², is the sRGB
   scale's 1.0, the light the roll-off ends at.
8. **sRGB encoding and quantisation.** `code = ⌊255 · sRGB(y) + 1/2⌋`,
   `sRGB(y) = 12.92 y` for `y ≤ 0.0031308` and `1.055 y^(1/2.4) − 0.055`
   above, which with `y ≤ 1` gives codes 0 to 255.

Reference values, from these formulas in `f64` (the value before rounding in
brackets): HLG 0% → 0; 50% → 137 (136.89); 75%, reference white → 229
(229.02), that is reference white sits at 90% of the sRGB range, near where
BT.2408 places SDR diffuse white; 100% → 255. PQ 203 cd/m² → 229 (228.99),
100 cd/m² → 186 (185.74). HLG 75% at −1 stop → 187 (186.91), at +1 stop →
250 (250.00). The BT.2020 green primary at HLG 75% → (0, 231, 0) and the red
primary → (239, 0, 0).

## Why this recipe

- **A fixed function of each pixel.** Tone mappers built for viewing (the
  `tonemap` filter's per-frame peak, libplacebo's peak detection, Dolby
  Vision's dynamic metadata) change a frame's mapping with its content. For a
  reconstruction the same surface must give the same pixel in every frame, so
  every adaptive element is left out: the constants above are the whole
  mapping, and one exposure is used for every frame of every video of a run.
- **Display light through the HLG OOTF.** BT.2100 defines an HLG signal's
  rendering on a display through the OOTF, and BT.2408's reference white
  (75% HLG = 203 cd/m²) is stated in display light, so the anchor and the
  roll-off, which are also stated in display light, apply to HLG and PQ
  alike. A 1000 cd/m² display is the nominal one BT.2100 and BT.2408 use.
- **BT.2390 EETF on max(R, G, B).** The EETF is the standard roll-off for
  mapping one HDR range into a smaller one: it leaves everything below the
  knee unchanged and joins a Hermite spline above it that reaches the target
  white with zero slope at the source white. Applying one factor from the
  largest channel keeps hue and keeps every channel within the target; a
  per-channel roll-off would turn bright saturated colours toward white.
- **Integer search for rounding.** Step 8 is done as a search in the 255
  linear values `sRGB⁻¹((k + 1/2)/255)`, computed once in `f64`: code `n` is
  the count of those thresholds at or below `y`. This rounds half up exactly
  as written and costs no `powf` per channel.

## Determinism and precision

All per-pixel arithmetic is `f32`. The input is 16-bit and the output 8-bit,
so the result needs about three significant digits. A pixel passes through a
fixed chain of about twenty operations with no sum over pixels; each adds at
most a few units of `2^-24` relative error, so the chain stays below `10^-5`
of the value, under a hundredth of an 8-bit level where one level is
smallest (about 0.4% of the linear value). Display light spans 0 to
10,000 cd/m² times at most `2^16`, inside `f32`'s range. An `f32` vector lane
holds twice the values of `f64`. The output of a pixel depends on nothing
but its codes and the options; it is the same at any thread count. `exp`
and `powf` come from the platform's maths library, so two platforms can
differ in the last bit of an intermediate value; that changes an output code
only when a value lands within about `10^-5` of a rounding threshold.

## Dolby Vision

A Dolby Vision video of profile 8.1 or 8.4 is a PQ or HLG base layer plus
per-frame reshaping metadata (the RPU). The base layer is converted here by
its own transfer, and the RPU is ignored: it is dynamic metadata, which the
recipe leaves out by design. Profile 5, whose base layer is not in a standard
colour space, is not converted correctly; it is a format of streaming
services, not of cameras.

## Testing

Rust unit tests in
[hdr_to_srgb/tests.rs](../../../crates/sfmtool-core/src/camera/image/hdr_to_srgb/tests.rs)
check the reference values above, computed by hand from the formulas
independently of the code, the out-of-gamut rule on the BT.2020 green and red
primaries, the exposure at −1 and +1 stop, that every grey code and every
single-primary ramp gives a non-decreasing output in its own channel, that
identical pixels anywhere in a frame give identical output, that one thread
and many give the same frame, the BT.2390 constants, and the errors.
`tests/test_extract_frames_command.py` converts tagged HLG and PQ videos end
to end ([extract-frames-command.md](../../cli/workspace/extract-frames-command.md#testing)).
