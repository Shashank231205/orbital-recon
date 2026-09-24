# Orbital Recon

Multi-modal satellite target detection and geospatial intelligence platform.

Detects military-relevant assets in satellite and drone imagery, converts each
detection to real-world coordinates, and lets an operator interrogate the
results in plain language.

---

## What it does

Upload an overhead scene. The system tiles it, enhances it according to the
sensor that produced it, runs oriented-box detection, merges duplicate
detections across tile seams, converts every surviving detection to WGS84, and
stores the result. Progress streams to the browser stage by stage. Once the run
completes, a language model answers questions grounded in what was actually
found.

| Capability | Implementation |
| --- | --- |
| Detection | YOLOv8-OBB, DOTA-pretrained, oriented bounding boxes |
| Large scenes | Sliding-window tiling with overlap and clamped edge origins |
| Duplicate removal | Per-class non-maximum suppression using exact rotated IoU |
| Multi-modality | CLAHE and percentile stretch for EO/IR, Lee speckle filter for SAR |
| Localisation | Affine transform plus CRS reprojection to WGS84 |
| Prioritisation | Asset taxonomy mapping detector labels to threat levels |
| Reporting | Gemini and Groq behind a failover router |
| Live feedback | Server-sent events per pipeline stage |

## Why oriented boxes

Overhead imagery has no canonical object orientation: a ship at anchor or an
aircraft on a taxiway can sit at any angle. An axis-aligned box around a
diagonal object is mostly background, which inflates the apparent overlap
between neighbouring objects and causes suppression to merge distinct targets
into one. Oriented boxes keep the geometry tight, and suppression compares them
with exact polygon intersection rather than an axis-aligned approximation.

On the sample harbour scene this matters concretely: 308 raw detections across
8 tiles reduce to 212 after suppression, and the surviving boxes carry rotations
from 0.34 to 1.32 radians that track individual hull headings.

---

## Architecture

```
orbital-recon/
├── backend/
│   └── src/orbital_recon/
│       ├── api/          FastAPI routes, dependencies, application factory
│       ├── core/         Settings, structured logging, exceptions, TLS
│       ├── db/           SQLAlchemy models and async sessions
│       ├── llm/          Provider interface, Gemini, Groq, failover router
│       ├── ml/
│       │   ├── preprocessing/   Tiling, modality-specific enhancement
│       │   ├── detectors/       OBB detector, asset taxonomy
│       │   ├── postprocess/     Oriented geometry, NMS, georeferencing
│       │   └── registry/        Model cache and device selection
│       ├── schemas/      Pydantic and dataclass contracts
│       └── services/     Pipeline orchestration, scenes, progress, intelligence
├── frontend/src/
│   ├── api/              Typed HTTP client
│   ├── features/         Upload, jobs, map, detections, query
│   ├── hooks/            Progress stream subscription
│   └── types/            Types mirroring the API schemas
├── docker/
└── samples/              Small scenes for a first run
```

Layering is enforced by direction of dependency: routes never touch the ORM
directly, services never touch HTTP, and the ML package knows nothing about the
database.

### Request flow

```
POST /scenes ─┬─> validate, store, create Scene + Job ──> 202 with job id
              └─> background task
                    │
                    ├─ load and enhance by modality
                    ├─ tile, infer per batch, report tile progress
                    ├─ merge tile detections with rotated NMS
                    ├─ reproject to WGS84 if the scene carries a CRS
                    └─ persist, then publish the terminal update
                              │
GET /jobs/{id}/stream <───────┘  server-sent events, one per stage
```

Inference runs in a worker thread so it never blocks the event loop. Progress
from that thread is marshalled back onto the loop and awaited, which keeps
updates strictly ordered.

---

## Running it

### Requirements

- Python 3.10 or newer
- Node 20 or newer
- An NVIDIA GPU is optional; the detector falls back to CPU automatically

### Backend

```bash
cd backend
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # Windows
# .venv/bin/python -m pip install -e ".[dev]"       # Linux and macOS

cp .env.example .env        # then add your API keys
.venv/Scripts/python -m uvicorn orbital_recon.api.app:app --port 8000
```

Interactive API documentation is served at `http://127.0.0.1:8000/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The dev server proxies `/api` to the backend, so
the browser stays on a single origin and the progress stream is not subject to
cross-origin restrictions.

### First run

Upload `samples/harbour-eo.jpg` for the pixel-space path. To see detections
placed on a map, generate the georeferenced counterpart first:

```bash
python scripts/make_geotiff_sample.py
```

Detector weights download automatically on first use, so the first analysis
takes longer than subsequent ones.

---

## Configuration

Settings are read from the environment or `backend/.env`. See `.env.example`
for the full list; the ones that matter most:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | none | Free key from [AI Studio](https://aistudio.google.com/apikey) |
| `GROQ_API_KEY` | none | Free key from [Groq Console](https://console.groq.com/keys) |
| `LLM_PROVIDER_ORDER` | `gemini,groq` | Priority order; the router fails over left to right |
| `DETECTION_DEVICE` | `auto` | `auto`, `cuda` or `cpu` |
| `CONFIDENCE_THRESHOLD` | `0.25` | Minimum detection confidence |
| `TILE_SIZE` | `640` | Tile edge in pixels |
| `TILE_OVERLAP` | `0.2` | Fraction shared between adjacent tiles |

Either provider key alone is enough to run. With neither, detection still works
and only the query endpoints report that no provider is available.

### Provider failover

Providers are tried in the configured order. One that fails is marked unhealthy
for a cooldown window so later requests skip it without paying its timeout
again, and is retried once the window expires. Each answer reports which
provider produced it, so a degraded primary is visible rather than silent.

Groq is typically an order of magnitude faster than Gemini on this workload. If
latency matters more than answer quality, set
`LLM_PROVIDER_ORDER=groq,gemini`.

---

## Testing

```bash
cd backend
.venv/Scripts/python -m pytest          # 119 tests
.venv/Scripts/python -m ruff check src tests

cd ../frontend
npm run typecheck
npm run build
```

Unit tests cover geometry, tiling, suppression, taxonomy, georeferencing,
persistence, progress fan-out, scene ingestion and provider failover.
Integration tests drive the real application over HTTP with a temporary
database, including the progress stream.

Two properties are asserted directly because they are easy to break and hard to
notice:

- every pixel of a scene appears in at least one tile
- progress updates arrive in non-decreasing order

---

## Notes on the hard parts

**Rotated IoU.** Overlap between oriented boxes is computed by clipping the two
polygons, not by comparing their axis-aligned bounds. The approximation
overestimates overlap badly for elongated diagonal objects, which is exactly the
shape most vessels and runways take.

**SAR speckle.** Speckle in synthetic-aperture radar is multiplicative, so
averaging filters erase edges along with the noise. The Lee filter weights each
pixel by local signal variance: it smooths homogeneous regions and backs off
near edges, preserving the hard structural returns that identify man-made
objects.

**Tile edges.** The final tile origin along each axis is clamped so the window
ends exactly at the image edge. Without that, the last tile is mostly padding
and objects near the right and bottom margins are systematically missed.

**Progress ordering.** Tile progress originates in a worker thread. Scheduling
those updates onto the event loop without awaiting them lets the pipeline
advance to a later stage before an earlier update is delivered, which reaches
the browser as a progress bar that jumps forward and then falls back.

**TLS interception.** Python verifies certificates against the `certifi`
bundle, which holds only public authorities. Machines running antivirus or
proxy HTTPS scanning present certificates signed by a locally installed root
that the bundle does not contain, so every outbound call fails to verify. The
application verifies against the operating system trust store instead, which
holds both. Verification stays fully enabled.

---

## Limitations

- The detector is pretrained on DOTA, an aerial survey dataset. Its classes are
  proxies for military assets, not trained on them: a "vessel" is any ship.
  Fine-tuning on domain data is the obvious next step.
- Detections are model output and carry uncertainty. The system reports
  confidence and never presents a detection as confirmed.
- Jobs run in-process. A deployment expecting sustained load should move them to
  a queue with a dedicated worker.
- SQLite suits a single-node demonstrator. Concurrent writers need PostgreSQL,
  and the ORM layer is already portable to it.
- Sample imagery is synthetic in its georeferencing: the transform was
  constructed, not acquired.

---

## Licence

Built with Ultralytics YOLO, which is AGPL-3.0. Any redistribution must comply
with those terms.
