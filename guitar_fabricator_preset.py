"""guitar_fabricator_preset.py
================================
GuitarFabricatorPreset — the fabricator recipe that produces a GuitarItem.

A fabricator preset is a static, catalog-registered specification for
making a particular instrument.  It carries:
  • catalog metadata (id, label, description, thumbnail colour, icon)
  • a fabricate() class method that builds and returns an inventory-ready
    GuitarItem in one call
  • a preview_mesh() class method that returns a lightweight silhouette
    suitable for the fabricator viewport before the full item is built
  • a catalog_entry() class method that returns the dict used in
    FabricatorWorkspace.SPECIAL_CATALOG

Usage (inside FabricatorStation or anywhere)::

    from guitar_fabricator_preset import GuitarFabricatorPreset

    item = GuitarFabricatorPreset.fabricate()
    # item is ready: item.in_inventory == True
    # pass to an InstrumentStation:
    station.load_item(item)

Custom configuration::

    cfg_overrides = {"fret": 5, "excitation": "a-pluck", "amr_backend": "cpu"}
    item = GuitarFabricatorPreset.fabricate(cfg_overrides)
"""
from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np

from guitar_item import GuitarItem, GuitarConfig, make_guitar_item
from guitar_geometry import guitar_outline


# ─────────────────────────────────────────────────────────────────────────────
# Preset
# ─────────────────────────────────────────────────────────────────────────────

class GuitarFabricatorPreset:
    """Catalog preset for a 6-string steel-string acoustic guitar.

    All methods are class methods; the preset carries no instance state.
    Obtain a built item via ``GuitarFabricatorPreset.fabricate()``.
    """

    #: Unique catalog identifier (must match key in SPECIAL_CATALOG).
    CATALOG_ID: str = "guitar_acoustic"

    #: Human-readable name shown in the fabricator list panel.
    LABEL: str = "Acoustic Guitar"

    #: Short description for the HUD tooltip.
    DESCRIPTION: str = (
        "6-string steel-string acoustic. "
        "FDTD Kirchhoff-plate body resonance (string_plate model). "
        "Plays any excitation (strum / a-pluck / rest)."
    )

    #: (R, G, B) swatch colour for the fabricator list panel icon.
    THUMBNAIL_COLOR: tuple[float, float, float] = (0.56, 0.30, 0.12)

    #: Unicode icon for the catalog tile.
    ICON: str = "🎸"

    # ── Build ─────────────────────────────────────────────────────────────────

    @classmethod
    def fabricate(cls,
                  config_overrides: Optional[Dict[str, Any]] = None
                  ) -> GuitarItem:
        """Build and return an inventory-ready GuitarItem.

        Parameters
        ----------
        config_overrides
            Optional dict of GuitarConfig field names → values to override
            before building the geometry.  Only top-level field names of
            GuitarConfig are recognised; unknown keys are silently ignored.

        Returns
        -------
        GuitarItem
            Freshly constructed item with ``item.in_inventory == True``.
            The item has all default part visibilities and no sim_sidecar
            data yet (that is populated by InstrumentStation after the FDTD
            build completes).
        """
        overrides = {k: v for k, v in (config_overrides or {}).items()
                     if hasattr(GuitarConfig, k) or k in GuitarConfig.__dataclass_fields__}
        cfg   = GuitarConfig(**overrides) if overrides else GuitarConfig()
        item  = make_guitar_item(cfg)
        item.acquire()
        return item

    # ── Preview ───────────────────────────────────────────────────────────────

    @classmethod
    def preview_mesh(cls) -> Dict[str, Any]:
        """Lightweight preview data for the fabricator 3-D viewport.

        Returns a dict with keys:
          outline   (N, 2) float32 — body silhouette polygon
          thumbnail_color  (3,) float32 — swatch colour
          label     str
        No full mesh is built; the caller can extrude or draw the outline
        directly in 2-D for a fast preview tile.
        """
        outline = guitar_outline(n_pts=64)
        return {
            "outline":         outline,
            "thumbnail_color": np.array(cls.THUMBNAIL_COLOR, np.float32),
            "label":           cls.LABEL,
        }

    # ── Catalog integration ───────────────────────────────────────────────────

    @classmethod
    def catalog_entry(cls) -> dict:
        """Return the SPECIAL_CATALOG entry dict for FabricatorWorkspace.

        The entry uses ``kind = "instrument"`` so FabricatorStation and
        RoomStation know this is an instrument preset rather than a light
        or portal.  The ``preset_class`` key gives the fabricator direct
        access to this class for calling ``fabricate()``.
        """
        return {
            "id":           cls.CATALOG_ID,
            "label":        cls.LABEL,
            "kind":         "instrument",
            "icon":         cls.ICON,
            "description":  cls.DESCRIPTION,
            "thumbnail_color": list(cls.THUMBNAIL_COLOR),
            "preset_class": cls,   # live reference for runtime fabrication
            "defaults": {
                "excitation":   "strum",
                "fret":         0,
                "fretless":     False,
                "amr_backend":  "cpu",
            },
        }
