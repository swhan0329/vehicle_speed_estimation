import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.io.video_source import _resolve_capture_source, open_video_source


class TestVideoSource(unittest.TestCase):
    def test_resolve_capture_source_uses_camera_index_for_numeric_string(self):
        self.assertEqual(_resolve_capture_source("0"), 0)
        self.assertEqual(_resolve_capture_source("-1"), -1)

    def test_resolve_capture_source_keeps_existing_numeric_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "0"
            file_path.write_text("dummy", encoding="utf-8")

            previous_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                self.assertEqual(_resolve_capture_source("0"), "0")
            finally:
                os.chdir(previous_cwd)

    def test_resolve_capture_source_keeps_non_numeric_source(self):
        self.assertEqual(
            _resolve_capture_source("samples/MNn9qKG2UFI_10s.mp4"),
            "samples/MNn9qKG2UFI_10s.mp4",
        )
        self.assertEqual(
            _resolve_capture_source("rtsp://127.0.0.1:8554/stream"),
            "rtsp://127.0.0.1:8554/stream",
        )

    def test_open_video_source_passes_resolved_source(self):
        with patch("app.io.video_source.cv.VideoCapture") as mock_capture:
            open_video_source("2")
            mock_capture.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
