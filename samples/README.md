# Sample scenes

Ten scenes covering every preprocessing path and several failure modes. Build
them with:

```bash
python scripts/fetch_samples.py
```

Only the first is downloaded. The rest are derived from it, so the set is
reproducible and no large binaries live in the repository. Derived scenes are
labelled as such in `manifest.json`: none of them is a real acquisition, and the
infrared and radar scenes in particular are approximations of what those sensors
produce, not samples from them.

## How to use these

For each scene, select the listed modality in the upload panel before analysing.
The modality is not a label: it chooses the preprocessing pipeline, and the
wrong choice produces visibly different results.

| # | Scene | Modality | What it exercises | Measured |
| --- | --- | --- | --- | --- |
| 01 | `01-harbour-vessels.jpg` | EO | Baseline. Dense targets, tile-seam merging. | 212 detections, 198 vessels |
| 02 | `02-harbour-georeferenced.tif` | EO | Georeferencing and the map view. | 212 detections, WGS84 coordinates |
| 03 | `03-harbour-infrared.png` | IR | Percentile stretch and CLAHE on a single band. | 45 detections, mostly reclassified |
| 04 | `04-harbour-radar.png` | SAR | Lee speckle filter. | 3 detections |
| 05 | `05-harbour-strip.png` | EO | Non-square tiling geometry. | 69 detections |
| 06 | `06-harbour-low-contrast.png` | EO | Contrast recovery from haze. | 205 detections |
| 07 | `07-harbour-rotated.png` | EO | Oriented boxes under scene rotation. | 215 detections |
| 08 | `08-harbour-quarter.png` | EO | Scene smaller than one tile. | 49 detections |
| 09 | `09-harbour-upscaled.png` | EO | Many tiles; visible staged progress. | 205 detections, ~1.6 s |
| 10 | `10-open-water.png` | EO | Negative control. | 0 detections |

Counts were measured on an RTX 3050 at the default 0.25 threshold. Expect small
variation with a different threshold or device.

## What each scene is worth looking at for

**01 — baseline.** The reference result. Note that boxes follow individual hull
angles rather than sitting square to the image: that is the oriented detector
working. Raw output is 308 detections across 8 tiles, reduced to 212 once
duplicates spanning tile seams are suppressed.

**02 — localisation.** The same imagery carrying a coordinate reference. The
view switches from a canvas to a map and every detection gains a latitude and
longitude. Select a target and compare its coordinates against the scene
transform; they agree to well under a metre.

**03 — infrared.** Detection count drops to about 45 and most survivors are
reclassified as vehicles. This is informative rather than a defect: stripping
colour removes the cues the model uses to tell a moored vessel from a parked
vehicle. It shows why a deployment against real thermal imagery needs a model
fine-tuned on it.

**04 — radar.** Only a few detections survive. The speckle applied here is
heavier than a real radar product, which usefully marks the limit of the
approach: the Lee filter preserves edges through multiplicative noise, but it
cannot restore texture that the noise has already destroyed. Compare against
scene 01 to see how much the model depends on that texture.

**05 — aspect ratio.** A wide, shallow crop producing a single row of tiles.
Confirms the tiling geometry does not assume a roughly square scene.

**06 — degraded contrast.** Haze compresses the scene toward mid grey, yet 205
of the baseline 212 detections survive, because CLAHE restores local contrast
before inference. Worth comparing side by side with scene 01.

**07 — rotation.** The scene is rotated 37 degrees. The count holds at 215 and
the box angles shift with it. Open a detection's detail card and read its
heading: it tracks the rotation rather than snapping to the image axes.

**08 — small scene.** Smaller than the 640-pixel tile window, so the pipeline
runs a single tile with no merging. The fastest path through the system.

**09 — large scene.** Roughly four megapixels, producing many tiles. The best
scene for watching the progress panel: stages advance visibly and the inference
stage reports tile counts as it works.

**10 — negative control.** Open water with all structure removed. It should
return nothing, and it does. A model that reports targets here is hallucinating,
which makes this the most important scene in the set despite being the least
interesting to look at.

## Suggested order

Run 01, then 02 to see localisation appear. Then 10 to confirm the system says
nothing when there is nothing. Then 03 and 04 to see the modality pipelines
diverge. The rest exercise geometry and scale.
