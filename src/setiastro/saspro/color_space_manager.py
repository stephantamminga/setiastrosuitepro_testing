"""Color-space helpers for display QImage tagging."""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QColorSpace, QImage

logger = logging.getLogger(__name__)

COLOR_SPACES = {
    "sRGB": QColorSpace.NamedColorSpace.SRgb,
    "DisplayP3": QColorSpace.NamedColorSpace.DisplayP3,
    "AdobeRGB": QColorSpace.NamedColorSpace.AdobeRgb,
    "ProPhotoRGB": QColorSpace.NamedColorSpace.ProPhotoRgb,
}

DEFAULT_COLOR_SPACE = "DisplayP3"
SETTINGS_KEY = "color_management/working_space"


def get_color_space_from_key(key: str) -> Optional[QColorSpace]:
    """Return a valid QColorSpace for a known key, otherwise None."""
    qt_key = COLOR_SPACES.get(key)
    if qt_key is None:
        logger.warning("Unknown color space key: %s", key)
        return None

    try:
        cs = QColorSpace(qt_key)
    except Exception:
        return None
    return cs if cs.isValid() else None


def get_working_color_space_key(settings: Optional[QSettings] = None) -> str:
    """Read working color-space key from settings with validation."""
    settings = settings or QSettings("SetiAstro", "SASpro")
    key = settings.value(SETTINGS_KEY, DEFAULT_COLOR_SPACE, type=str)
    return key if key in COLOR_SPACES else DEFAULT_COLOR_SPACE


class ColorSpaceManager:
    """Singleton accessor for working QColorSpace and image tagging."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._settings = QSettings("SetiAstro", "SASpro")
        return cls._instance

    def get_working_space_key(self) -> str:
        return get_working_color_space_key(self._settings)

    def get_working_space_qcolor_space(self) -> Optional[QColorSpace]:
        return get_color_space_from_key(self.get_working_space_key())

    def tag_qimage(self, img: QImage) -> QImage:
        if img is None:
            return img
        cs = self.get_working_space_qcolor_space()
        if cs is not None:
            try:
                img.setColorSpace(cs)
            except Exception:
                pass
        return img


def tag_qimage_with_working_color_space(img: QImage) -> QImage:
    """Tag a display QImage with the current working color space."""
    return ColorSpaceManager().tag_qimage(img)
