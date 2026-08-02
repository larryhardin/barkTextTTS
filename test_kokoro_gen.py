import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import kokoro_gen


class KokoroGenTests(unittest.TestCase):
    def test_parser_accepts_cuda_device_aliases(self) -> None:
        parser = kokoro_gen.build_arg_parser()

        args_dash = parser.parse_args(["--cuda-device", "1"])
        self.assertEqual(args_dash.cuda_device, "1")

        args_underscore = parser.parse_args(["--cuda_device", "0"])
        self.assertEqual(args_underscore.cuda_device, "0")

    def test_generate_kokoro_tts_concatenates_pipeline_audio(self) -> None:
        fake_pipeline = MagicMock(
            return_value=[
                ("first", "f er s t", np.array([0.25, -0.5], dtype=np.float32)),
                ("second", "s eh k ah n d", np.array([0.1], dtype=np.float32)),
            ]
        )

        with patch("kokoro_gen.load_kokoro_pipeline", return_value=fake_pipeline):
            audio = kokoro_gen.generate_kokoro_tts(
                "Hello world",
                voice_profile="af_heart",
                lang_code="a",
                speed=1.0,
                return_audio_array=True,
            )

        self.assertIsNotNone(audio)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.shape[0], 3)
        self.assertAlmostEqual(float(np.max(np.abs(audio))), 1.0)
        fake_pipeline.assert_called_once_with("Hello world", voice="af_heart", speed=1.0, split_pattern=r"\n+")

    def test_blank_lines_add_silence_and_skip_generation(self) -> None:
        segments = ["first line", "", "third line"]

        with patch(
            "kokoro_gen.generate_kokoro_tts",
            side_effect=[np.array([0.1, 0.2], dtype=np.float32), np.array([0.3, 0.4], dtype=np.float32)],
        ) as mocked:
            combined_audio = kokoro_gen.process_segments_to_audio(segments, voice_profile="af_heart", lang_code="a", speed=1.0)

        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(combined_audio.shape[0], 24004)
        self.assertTrue(np.allclose(combined_audio[2:24002], 0.0))


if __name__ == "__main__":
    unittest.main()