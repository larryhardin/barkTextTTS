import os
import tempfile
import unittest
import uuid
from unittest.mock import patch

import numpy as np

import cuda_gen


class CudaGenTests(unittest.TestCase):
    def test_parser_accepts_cuda_device_aliases(self) -> None:
        parser = cuda_gen.build_arg_parser()

        args_dash = parser.parse_args(["--cuda-device", "1"])
        self.assertEqual(args_dash.cuda_device, "1")

        args_underscore = parser.parse_args(["--cuda_device", "0"])
        self.assertEqual(args_underscore.cuda_device, "0")

    def test_generate_output_filename_returns_uuid_wav_name(self) -> None:
        result = cuda_gen.generate_output_filename()

        self.assertTrue(result.endswith(".wav"))
        parsed = uuid.UUID(result[:-4])
        self.assertEqual(str(parsed), result[:-4])

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
