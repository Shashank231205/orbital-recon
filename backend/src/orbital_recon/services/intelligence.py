"""Natural-language reporting over detection results.

An operator asks questions in plain language; the model answers from a
structured summary of what the pipeline actually found. The summary is computed
here rather than handing raw rows to the model, for three reasons: a dense scene
would exceed the context window, aggregates are what the questions are usually
about, and grounding the answer in fixed numbers limits invention.
"""

from collections import Counter
from dataclasses import dataclass

from orbital_recon.core.logging import get_logger
from orbital_recon.db.models import Detection, Job, Scene
from orbital_recon.llm.base import ChatMessage, CompletionResult
from orbital_recon.llm.router import LLMRouter
from orbital_recon.schemas.enums import ThreatLevel

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are an imagery analyst assistant for a geospatial intelligence platform.

Answer strictly from the scene summary provided. Follow these rules:
- Never invent detections, coordinates or counts that are not in the summary.
- If the summary does not contain the answer, say so plainly.
- Report counts and coordinates exactly as given.
- Be concise and factual. Use the vocabulary of imagery analysis.
- Note that detections are model output and carry uncertainty.
"""

# Caps the coordinate list so a dense scene cannot crowd out the question.
_MAX_LISTED_DETECTIONS = 40


@dataclass(frozen=True, slots=True)
class SceneSummary:
    """Aggregated findings for one analysis job."""

    job_id: int
    scene_name: str
    is_georeferenced: bool
    total_detections: int
    by_class: dict[str, int]
    by_threat: dict[str, int]
    mean_confidence: float
    highest_confidence: float

    def to_prompt_block(self, detections: list[Detection]) -> str:
        """Render the summary as the grounding context for the model."""
        lines = [
            f"Scene: {self.scene_name}",
            f"Job ID: {self.job_id}",
            f"Georeferenced: {'yes' if self.is_georeferenced else 'no'}",
            f"Total detections: {self.total_detections}",
            "",
            "Detections by asset class:",
        ]
        lines += [f"  {name}: {count}" for name, count in sorted(self.by_class.items())]

        lines += ["", "Detections by threat level:"]
        lines += [f"  {level}: {count}" for level, count in sorted(self.by_threat.items())]

        lines += [
            "",
            f"Mean confidence: {self.mean_confidence:.3f}",
            f"Highest confidence: {self.highest_confidence:.3f}",
        ]

        if detections:
            shown = detections[:_MAX_LISTED_DETECTIONS]
            lines += ["", f"Highest-confidence detections (showing {len(shown)}):"]
            for index, detection in enumerate(shown, start=1):
                position = (
                    f"{detection.latitude:.5f}, {detection.longitude:.5f}"
                    if detection.latitude is not None and detection.longitude is not None
                    else f"pixel {detection.cx:.0f}, {detection.cy:.0f}"
                )
                lines.append(
                    f"  {index}. {detection.display_name} "
                    f"({detection.threat_level}) "
                    f"conf={detection.confidence:.3f} at {position}"
                )
            if len(detections) > len(shown):
                lines.append(f"  ... and {len(detections) - len(shown)} more")

        return "\n".join(lines)


def build_summary(job: Job, scene: Scene, detections: list[Detection]) -> SceneSummary:
    """Aggregate a job's detections into a summary."""
    confidences = [detection.confidence for detection in detections]

    return SceneSummary(
        job_id=job.id,
        scene_name=scene.filename,
        is_georeferenced=scene.is_georeferenced,
        total_detections=len(detections),
        by_class=dict(Counter(detection.display_name for detection in detections)),
        by_threat=dict(Counter(str(detection.threat_level) for detection in detections)),
        mean_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
        highest_confidence=max(confidences, default=0.0),
    )


def priority_ordered(detections: list[Detection]) -> list[Detection]:
    """Sort detections by operational priority, then confidence.

    Threat level dominates because an operator triaging a scene cares about a
    moderately confident aircraft before a highly confident swimming pool.
    """
    rank = {
        ThreatLevel.HIGH: 0,
        ThreatLevel.MEDIUM: 1,
        ThreatLevel.LOW: 2,
        ThreatLevel.UNKNOWN: 3,
    }
    return sorted(
        detections,
        key=lambda d: (rank.get(ThreatLevel(d.threat_level), 3), -d.confidence),
    )


class IntelligenceService:
    """Answers operator questions and drafts briefs from detection results."""

    def __init__(self, router: LLMRouter) -> None:
        self._router = router

    async def answer(
        self,
        question: str,
        job: Job,
        scene: Scene,
        detections: list[Detection],
    ) -> CompletionResult:
        """Answer a question grounded in one job's findings."""
        summary = build_summary(job, scene, detections)
        context = summary.to_prompt_block(priority_ordered(detections))

        return await self._router.complete(
            [
                ChatMessage(role="system", content=_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=f"Scene summary:\n{context}\n\nQuestion: {question}",
                ),
            ],
            temperature=0.2,
        )

    async def brief(
        self, job: Job, scene: Scene, detections: list[Detection]
    ) -> CompletionResult:
        """Draft a short intelligence brief for a completed job."""
        summary = build_summary(job, scene, detections)
        context = summary.to_prompt_block(priority_ordered(detections))

        return await self._router.complete(
            [
                ChatMessage(role="system", content=_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=(
                        f"Scene summary:\n{context}\n\n"
                        "Write a brief of at most 200 words covering: what was "
                        "detected, which assets warrant attention first, and any "
                        "limitations a reader should keep in mind."
                    ),
                ),
            ],
            temperature=0.3,
            max_tokens=512,
        )
