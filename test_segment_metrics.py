import json
import os
import tempfile
import unittest

from segment_metrics import build_segment_metrics


class SegmentMetricsTests(unittest.TestCase):
    def test_build_segment_metrics_computes_expected_values(self):
        data = {
            "segments": [
                {"id": 0, "start": 0.0, "end": 4.3, "temperature": 0.0},
                {"id": 1, "start": 4.3, "end": 9.94, "temperature": 0.1},
                {"id": 2, "start": 10.26, "end": 14.14, "temperature": 0.1},
            ]
        }

        result = build_segment_metrics(data)

        self.assertEqual(result[0]["duration"], 4.3)
        self.assertEqual(result[0]["temperature_change"], None)
        self.assertEqual(result[0]["time_since_previous_end"], None)

        self.assertEqual(result[1]["duration"], 5.64)
        self.assertEqual(result[1]["temperature_change"], 0.1)
        self.assertEqual(result[1]["time_since_previous_end"], 0.0)

        self.assertEqual(result[2]["duration"], 3.88)
        self.assertEqual(result[2]["temperature_change"], 0.0)
        self.assertEqual(result[2]["time_since_previous_end"], 0.32)


if __name__ == "__main__":
    unittest.main()
