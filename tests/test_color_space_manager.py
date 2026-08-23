from __future__ import annotations

import os

import pytest

pytest.importorskip("PyQt6")
from setiastro.saspro import color_space_manager as csm


@pytest.fixture
def qapp():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    return app


def test_get_working_color_space_key_falls_back_for_invalid_value(tmp_path):
    from PyQt6.QtCore import QSettings

    settings = QSettings(str(tmp_path / "color.ini"), QSettings.Format.IniFormat)
    settings.setValue(csm.SETTINGS_KEY, "InvalidSpace")

    assert csm.get_working_color_space_key(settings) == csm.DEFAULT_COLOR_SPACE


def test_tag_qimage_with_working_color_space_sets_valid_color_space(monkeypatch, qapp):
    from PyQt6.QtGui import QImage

    cs = csm.get_color_space_from_key("sRGB")
    assert cs is not None and cs.isValid()

    monkeypatch.setattr(csm.ColorSpaceManager, "get_working_space_qcolor_space", lambda self: cs)

    img = QImage(8, 8, QImage.Format.Format_RGB888)
    out = csm.tag_qimage_with_working_color_space(img)

    assert out is img
    assert out.colorSpace().isValid()
