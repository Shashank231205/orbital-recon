"""End-to-end tests over the HTTP API.

These exercise the real application with a temporary database and upload
directory. Detection itself is replaced by a stub so the suite stays fast and
deterministic; the detector has its own unit coverage.
"""

import io
import json
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image

from orbital_recon.api.app import create_app
from orbital_recon.core.config import Settings
from orbital_recon.db import session as session_module
from orbital_recon.ml.postprocess.geometry import OrientedBox
from orbital_recon.schemas.detection import RawDetection

API = "/api/v1"


def png_bytes(width: int = 256, height: int = 256) -> bytes:
    """A small valid PNG payload."""
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="test",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        data_dir=tmp_path,
        upload_dir=tmp_path / "raw",
        model_dir=tmp_path / "models",
        gemini_api_key=None,
        groq_api_key=None,
    )


@pytest_asyncio.fixture
async def client(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncClient]:
    """An HTTP client bound to the app, with detection stubbed out."""

    def fake_detect(self, image, on_progress=None):  # type: ignore[no-untyped-def]
        if on_progress is not None:
            on_progress(1, 1)
        return [
            (
                RawDetection(
                    box=OrientedBox(50.0, 60.0, 20.0, 10.0, 0.25),
                    class_name="ship",
                    confidence=0.91,
                ),
                0,
                0,
            ),
            (
                RawDetection(
                    box=OrientedBox(150.0, 40.0, 12.0, 8.0, 0.0),
                    class_name="small vehicle",
                    confidence=0.42,
                ),
                0,
                0,
            ),
        ]

    monkeypatch.setattr(
        "orbital_recon.ml.detectors.obb_detector.ObbDetector.__init__",
        lambda self, **kwargs: None,
    )
    monkeypatch.setattr(
        "orbital_recon.ml.detectors.obb_detector.ObbDetector.detect", fake_detect
    )

    # The engine is module-level state shared across the process.
    await session_module.dispose_engine()

    app = create_app(settings)
    async with (
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client,
        app.router.lifespan_context(app),
    ):
        yield http_client

    await session_module.dispose_engine()


async def upload(client: AsyncClient, filename: str = "scene.png") -> dict:
    """Upload a scene and return the parsed response."""
    response = await client.post(
        f"{API}/scenes",
        files={"file": (filename, png_bytes(), "image/png")},
        data={"modality": "eo"},
    )
    assert response.status_code == 202, response.text
    return response.json()


async def collect_stream(client: AsyncClient, job_id: int) -> list[dict]:
    """Return every progress update a job emits, in arrival order."""
    updates: list[dict] = []
    async with client.stream("GET", f"{API}/jobs/{job_id}/stream") as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue

            update = json.loads(line[6:])
            updates.append(update)
            if update["status"] in ("completed", "failed"):
                return updates
    raise AssertionError("stream ended before the job reached a terminal state")


async def wait_for_completion(client: AsyncClient, job_id: int) -> dict:
    """Follow the progress stream until the job reaches a terminal state."""
    return (await collect_stream(client, job_id))[-1]


class TestHealth:
    async def test_reports_service_state(self, client: AsyncClient) -> None:
        body = (await client.get(f"{API}/health")).json()

        assert body["status"] == "ok"
        assert body["detector_loaded"] is False
        assert {p["name"] for p in body["llm_providers"]} == {"gemini", "groq"}

    async def test_providers_unconfigured_without_keys(self, client: AsyncClient) -> None:
        body = (await client.get(f"{API}/health")).json()
        assert all(not p["configured"] for p in body["llm_providers"])


class TestUpload:
    async def test_accepts_image_and_queues_job(self, client: AsyncClient) -> None:
        body = await upload(client)

        assert body["scene"]["filename"] == "scene.png"
        assert body["scene"]["is_georeferenced"] is False
        assert body["job"]["status"] == "pending"

    async def test_records_true_dimensions(self, client: AsyncClient) -> None:
        response = await client.post(
            f"{API}/scenes",
            files={"file": ("odd.png", png_bytes(321, 123), "image/png")},
            data={"modality": "eo"},
        )
        scene = response.json()["scene"]

        assert (scene["width"], scene["height"]) == (321, 123)

    async def test_rejects_unsupported_format(self, client: AsyncClient) -> None:
        response = await client.post(
            f"{API}/scenes",
            files={"file": ("notes.txt", b"hello", "text/plain")},
            data={"modality": "eo"},
        )

        assert response.status_code == 415
        assert response.json()["code"] == "unsupported_media"

    async def test_rejects_empty_upload(self, client: AsyncClient) -> None:
        response = await client.post(
            f"{API}/scenes",
            files={"file": ("empty.png", b"", "image/png")},
            data={"modality": "eo"},
        )

        assert response.status_code == 422


class TestJobLifecycle:
    async def test_stream_reports_progress_to_completion(self, client: AsyncClient) -> None:
        job_id = (await upload(client))["job"]["id"]

        final = await wait_for_completion(client, job_id)

        assert final["status"] == "completed"
        assert final["progress"] == 1.0
        assert final["detection_count"] == 2

    async def test_job_record_reflects_completion(self, client: AsyncClient) -> None:
        job_id = (await upload(client))["job"]["id"]
        await wait_for_completion(client, job_id)

        body = (await client.get(f"{API}/jobs/{job_id}")).json()

        assert body["status"] == "completed"
        assert body["stage"] == "done"
        assert body["duration_ms"] is not None

    async def test_progress_never_moves_backwards(self, client: AsyncClient) -> None:
        """Stage updates must arrive in order.

        Inference reports progress from a worker thread; if those updates are
        scheduled without being awaited, a later stage can overtake them and
        the client sees the bar jump back.
        """
        job_id = (await upload(client))["job"]["id"]

        updates = await collect_stream(client, job_id)
        values = [update["progress"] for update in updates]

        assert values == sorted(values), f"progress went backwards: {values}"

    async def test_unknown_job_returns_not_found(self, client: AsyncClient) -> None:
        response = await client.get(f"{API}/jobs/99999")

        assert response.status_code == 404
        assert response.json()["code"] == "not_found"


class TestDetections:
    async def test_lists_detections_ordered_by_confidence(self, client: AsyncClient) -> None:
        job_id = (await upload(client))["job"]["id"]
        await wait_for_completion(client, job_id)

        body = (await client.get(f"{API}/jobs/{job_id}/detections")).json()

        assert body["total"] == 2
        confidences = [item["confidence"] for item in body["items"]]
        assert confidences == sorted(confidences, reverse=True)

    async def test_box_corners_are_returned_for_drawing(self, client: AsyncClient) -> None:
        job_id = (await upload(client))["job"]["id"]
        await wait_for_completion(client, job_id)

        item = (await client.get(f"{API}/jobs/{job_id}/detections")).json()["items"][0]

        assert len(item["box"]["corners"]) == 4

    async def test_confidence_filter_applies(self, client: AsyncClient) -> None:
        job_id = (await upload(client))["job"]["id"]
        await wait_for_completion(client, job_id)

        body = (
            await client.get(f"{API}/jobs/{job_id}/detections", params={"min_confidence": 0.5})
        ).json()

        assert body["total"] == 1
        assert body["items"][0]["class_name"] == "Vessel"

    async def test_threat_filter_applies(self, client: AsyncClient) -> None:
        job_id = (await upload(client))["job"]["id"]
        await wait_for_completion(client, job_id)

        body = (
            await client.get(f"{API}/jobs/{job_id}/detections", params={"threat_level": "high"})
        ).json()

        assert body["total"] == 1

    async def test_non_georeferenced_scene_has_no_coordinates(
        self, client: AsyncClient
    ) -> None:
        """A plain PNG carries no CRS, so detections stay in pixel space."""
        job_id = (await upload(client))["job"]["id"]
        await wait_for_completion(client, job_id)

        items = (await client.get(f"{API}/jobs/{job_id}/detections")).json()["items"]

        assert all(item["centroid"] is None for item in items)


class TestSummary:
    async def test_aggregates_by_class_and_threat(self, client: AsyncClient) -> None:
        job_id = (await upload(client))["job"]["id"]
        await wait_for_completion(client, job_id)

        body = (await client.get(f"{API}/jobs/{job_id}/summary")).json()

        assert body["total_detections"] == 2
        assert body["by_threat"] == {"high": 1, "medium": 1}
        assert body["mean_confidence"] == pytest.approx(0.665, abs=1e-3)


class TestIntelligence:
    async def test_query_reports_unavailable_without_keys(self, client: AsyncClient) -> None:
        """With no provider configured the failure is explicit, not a crash."""
        job_id = (await upload(client))["job"]["id"]
        await wait_for_completion(client, job_id)

        response = await client.post(
            f"{API}/jobs/{job_id}/query", json={"question": "What was detected?"}
        )

        assert response.status_code == 503
        assert response.json()["code"] == "llm_unavailable"

    async def test_rejects_empty_question(self, client: AsyncClient) -> None:
        job_id = (await upload(client))["job"]["id"]

        response = await client.post(f"{API}/jobs/{job_id}/query", json={"question": ""})

        assert response.status_code == 422
