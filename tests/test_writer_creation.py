import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.pipeline import _create_writer


class _FakeWriter:
    def __init__(self, opened: bool) -> None:
        self._opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def release(self) -> None:
        self.released = True


class TestWriterCreation(unittest.TestCase):
    def test_create_writer_falls_back_and_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "out.mp4"
            writers = [_FakeWriter(False), _FakeWriter(False), _FakeWriter(True)]
            calls: list[tuple[str, int, float, tuple[int, int]]] = []

            def fake_video_writer(
                path: str,
                fourcc: int,
                fps: float,
                size: tuple[int, int],
            ) -> _FakeWriter:
                calls.append((path, fourcc, fps, size))
                return writers[len(calls) - 1]

            with (
                patch("app.pipeline.cv.VideoWriter_fourcc", side_effect=[1, 2, 3]),
                patch("app.pipeline.cv.VideoWriter", side_effect=fake_video_writer),
            ):
                writer = _create_writer(str(output_path), (1280, 720), 30.0)

            self.assertIs(writer, writers[2])
            self.assertEqual(len(calls), 3)
            self.assertTrue(writers[0].released)
            self.assertTrue(writers[1].released)
            self.assertTrue(output_path.parent.exists())

    def test_create_writer_returns_none_when_all_codecs_fail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "out.mp4"
            writers = [_FakeWriter(False), _FakeWriter(False), _FakeWriter(False), _FakeWriter(False)]

            with (
                patch("app.pipeline.cv.VideoWriter_fourcc", side_effect=[1, 2, 3, 4]),
                patch("app.pipeline.cv.VideoWriter", side_effect=writers),
                patch("builtins.print") as mock_print,
            ):
                writer = _create_writer(str(output_path), (640, 480), 30.0)

            self.assertIsNone(writer)
            self.assertTrue(all(item.released for item in writers))
            printed = " ".join(str(arg) for arg in mock_print.call_args_list[-1][0])
            self.assertIn("attempted codecs", printed)


if __name__ == "__main__":
    unittest.main()
