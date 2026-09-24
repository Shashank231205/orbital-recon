# Sample scenes

Small scenes for exercising the pipeline without downloading a dataset.

| File | Format | Use |
| --- | --- | --- |
| `harbour-eo.jpg` | JPEG, 1920x1080 | Pixel-space path. No georeferencing, so detections are drawn over the image on a canvas. |
| `harbour-georeferenced.tif` | GeoTIFF, EPSG:32643 | Geographic path. Detections carry WGS84 coordinates and appear on the map. |

The GeoTIFF is not committed, because it is an order of magnitude larger than
the JPEG and carries no information its transform does not already describe.
Generate it once:

```bash
python scripts/make_geotiff_sample.py
```

It places the same imagery near Mumbai at 0.5 m per pixel. The coordinates are
internally consistent but synthetic: this is not a real acquisition.

Expect roughly 210 detections at the default 0.25 confidence threshold, mostly
vessels with a handful of harbour structures.
