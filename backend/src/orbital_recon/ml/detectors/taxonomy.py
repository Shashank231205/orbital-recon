"""Mapping from detector output classes to operational asset categories.

The pretrained DOTA weights emit aerial survey labels such as ``plane`` or
``large vehicle``. Those names are not how an analyst thinks about a scene, so
they are translated into asset categories carrying an operational priority.
Classes with no military relevance are mapped to LOW rather than dropped, since
a harbour or bridge is still useful context for interpreting what sits near it.
"""

from dataclasses import dataclass

from orbital_recon.schemas.enums import ThreatLevel


@dataclass(frozen=True, slots=True)
class AssetClass:
    """An operational asset category derived from a raw detector label."""

    label: str
    display_name: str
    threat_level: ThreatLevel


_DOTA_ASSET_MAP: dict[str, AssetClass] = {
    "plane": AssetClass("aircraft", "Aircraft", ThreatLevel.HIGH),
    "helicopter": AssetClass("helicopter", "Helicopter", ThreatLevel.HIGH),
    "ship": AssetClass("vessel", "Vessel", ThreatLevel.HIGH),
    "large vehicle": AssetClass("heavy_vehicle", "Heavy Vehicle", ThreatLevel.MEDIUM),
    "small vehicle": AssetClass("light_vehicle", "Light Vehicle", ThreatLevel.MEDIUM),
    "storage tank": AssetClass("storage_tank", "Storage Tank", ThreatLevel.MEDIUM),
    "harbor": AssetClass("harbour", "Harbour", ThreatLevel.LOW),
    "bridge": AssetClass("bridge", "Bridge", ThreatLevel.LOW),
    "roundabout": AssetClass("roundabout", "Roundabout", ThreatLevel.LOW),
    "tennis court": AssetClass("tennis_court", "Tennis Court", ThreatLevel.LOW),
    "basketball court": AssetClass("basketball_court", "Basketball Court", ThreatLevel.LOW),
    "ground track field": AssetClass("track_field", "Track Field", ThreatLevel.LOW),
    "soccer ball field": AssetClass("sports_field", "Sports Field", ThreatLevel.LOW),
    "baseball diamond": AssetClass("sports_field", "Sports Field", ThreatLevel.LOW),
    "swimming pool": AssetClass("swimming_pool", "Swimming Pool", ThreatLevel.LOW),
}

_UNKNOWN = AssetClass("unknown", "Unknown", ThreatLevel.UNKNOWN)


def resolve_asset_class(detector_label: str) -> AssetClass:
    """Translate a raw detector label into an asset category.

    Unrecognised labels resolve to an explicit unknown category so that a model
    swap surfaces as visibly unclassified detections rather than silent drops.
    """
    normalised = detector_label.lower().strip().replace("-", " ").replace("_", " ")
    return _DOTA_ASSET_MAP.get(normalised, _UNKNOWN)


def known_labels() -> tuple[str, ...]:
    """Asset labels this taxonomy can produce, for API documentation and filters."""
    return tuple(sorted({asset.label for asset in _DOTA_ASSET_MAP.values()}))
