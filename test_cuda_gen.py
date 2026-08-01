import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import cuda_gen


class CudaGenTests(unittest.TestCase):
    def test_concatenate_audio_segments_returns_combined_array(self) -> None:
        chunks = [
            np.array([0.1, -0.2], dtype=np.float32),
            np.array([0.3, 0.4], dtype=np.float32),
        ]

        result = cuda_gen.concatenate_audio_segments(chunks)

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32))

    def test_blank_lines_add_silence_and_skip_tts_generation(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
            handle.write("first line\n\nthird line\n")
            temp_path = handle.name

        try:
            segments = cuda_gen.load_segments_from_file(temp_path)
            with patch("cuda_gen.generate_high_quality_tts", side_effect=[np.array([0.1, 0.2], dtype=np.float32), np.array([0.3, 0.4], dtype=np.float32)]) as mocked:
                combined_audio = cuda_gen.process_segments_to_audio(segments, voice_profile="v2/en_speaker_7")

            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(combined_audio.shape[0], 6)
            self.assertTrue(np.allclose(combined_audio[2:4], 0.0))
        finally:
            os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
