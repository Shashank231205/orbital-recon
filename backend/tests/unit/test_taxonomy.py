"""Tests for the asset class taxonomy."""

import pytest

from orbital_recon.ml.detectors.taxonomy import known_labels, resolve_asset_class
from orbital_recon.schemas.enums import ThreatLevel

# The exact class names emitted by the pretrained DOTA OBB weights. Pinned here
# so that swapping weights fails loudly in tests instead of silently mapping
# every detection to unknown.
DOTA_CLASS_NAMES = (
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
)


class TestResolveAssetClass:
    @pytest.mark.parametrize("label", DOTA_CLASS_NAMES)
    def test_every_model_class_is_mapped(self, label: str) -> None:
        assert resolve_asset_class(label).threat_level is not ThreatLevel.UNKNOWN

    def test_high_priority_assets(self) -> None:
        for label in ("plane", "helicopter", "ship"):
            assert resolve_asset_class(label).threat_level is ThreatLevel.HIGH

    def test_vehicles_are_medium_priority(self) -> None:
        assert resolve_asset_class("large vehicle").label == "heavy_vehicle"
        assert resolve_asset_class("small vehicle").label == "light_vehicle"

    def test_separator_and_case_insensitive(self) -> None:
        """Hyphen, underscore and space spellings resolve identically."""
        expected = resolve_asset_class("large vehicle")
        for variant in ("large-vehicle", "Large_Vehicle", "  LARGE VEHICLE  "):
            assert resolve_asset_class(variant) == expected

    def test_unrecognised_label_is_explicitly_unknown(self) -> None:
        asset = resolve_asset_class("submarine")
        assert asset.label == "unknown"
        assert asset.threat_level is ThreatLevel.UNKNOWN

    def test_known_labels_are_sorted_and_unique(self) -> None:
        labels = known_labels()
        assert list(labels) == sorted(set(labels))
