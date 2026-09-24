"""Tests for the persistence layer."""

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from orbital_recon.db.models import Base, Detection, Job, Scene
from orbital_recon.schemas.enums import JobStage, JobStatus, Modality, ThreatLevel


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as active:
        yield active

    await engine.dispose()


async def make_scene(session: AsyncSession) -> Scene:
    scene = Scene(
        filename="scene.tif",
        stored_path="/data/raw/scene.tif",
        modality=Modality.ELECTRO_OPTICAL,
        width=2048,
        height=2048,
        size_bytes=1024,
        crs="EPSG:32643",
        is_georeferenced=True,
    )
    session.add(scene)
    await session.commit()
    return scene


class TestScene:
    async def test_persists_with_defaults(self, session: AsyncSession) -> None:
        scene = await make_scene(session)
        assert scene.id is not None
        assert scene.created_at is not None

    async def test_modality_round_trips(self, session: AsyncSession) -> None:
        scene = await make_scene(session)
        fetched = await session.get(Scene, scene.id)
        assert fetched is not None
        assert fetched.modality == Modality.ELECTRO_OPTICAL


class TestJob:
    async def test_defaults_to_pending_and_queued(self, session: AsyncSession) -> None:
        scene = await make_scene(session)
        job = Job(scene_id=scene.id, confidence_threshold=0.25)
        session.add(job)
        await session.commit()

        assert job.status == JobStatus.PENDING
        assert job.stage == JobStage.QUEUED
        assert job.progress == 0.0

    async def test_updated_at_advances_on_change(self, session: AsyncSession) -> None:
        scene = await make_scene(session)
        job = Job(scene_id=scene.id, confidence_threshold=0.25)
        session.add(job)
        await session.commit()
        first = job.updated_at

        job.stage = JobStage.INFERENCE
        job.progress = 0.5
        await session.commit()

        assert job.updated_at >= first


class TestDetection:
    async def test_stores_pixel_and_geographic_coordinates(
        self, session: AsyncSession
    ) -> None:
        scene = await make_scene(session)
        job = Job(scene_id=scene.id, confidence_threshold=0.25)
        session.add(job)
        await session.commit()

        session.add(
            Detection(
                job_id=job.id,
                class_label="vessel",
                display_name="Vessel",
                threat_level=ThreatLevel.HIGH,
                confidence=0.91,
                cx=100.0,
                cy=200.0,
                width=40.0,
                height=12.0,
                angle=0.35,
                latitude=13.02,
                longitude=76.61,
            )
        )
        await session.commit()

        stored = (await session.execute(select(Detection))).scalar_one()
        assert stored.threat_level == ThreatLevel.HIGH
        assert stored.latitude == pytest.approx(13.02)

    async def test_geographic_fields_optional(self, session: AsyncSession) -> None:
        """Detections from a non-georeferenced scene carry no position."""
        scene = await make_scene(session)
        job = Job(scene_id=scene.id, confidence_threshold=0.25)
        session.add(job)
        await session.commit()

        session.add(
            Detection(
                job_id=job.id,
                class_label="aircraft",
                display_name="Aircraft",
                threat_level=ThreatLevel.HIGH,
                confidence=0.8,
                cx=1.0,
                cy=2.0,
                width=3.0,
                height=4.0,
                angle=0.0,
            )
        )
        await session.commit()

        stored = (await session.execute(select(Detection))).scalar_one()
        assert stored.latitude is None
        assert stored.footprint_geojson is None


class TestCascades:
    async def test_deleting_scene_removes_jobs_and_detections(
        self, session: AsyncSession
    ) -> None:
        scene = await make_scene(session)
        job = Job(scene_id=scene.id, confidence_threshold=0.25)
        session.add(job)
        await session.commit()

        session.add(
            Detection(
                job_id=job.id,
                class_label="vessel",
                display_name="Vessel",
                threat_level=ThreatLevel.HIGH,
                confidence=0.9,
                cx=1.0,
                cy=1.0,
                width=1.0,
                height=1.0,
                angle=0.0,
            )
        )
        await session.commit()

        await session.delete(scene)
        await session.commit()

        assert (await session.execute(select(func.count(Job.id)))).scalar_one() == 0
        assert (await session.execute(select(func.count(Detection.id)))).scalar_one() == 0
