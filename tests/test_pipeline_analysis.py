import unittest
from unittest.mock import patch

import pandas as pd

from app.core.config import settings
from app.services.pipeline import _load_analysis_sample


class AnalysisSampleLoadingTests(unittest.TestCase):
    def test_load_analysis_sample_limits_rows(self):
        chunks = [
            pd.DataFrame({"value": [1, 2, 3]}),
            pd.DataFrame({"value": [4, 5, 6, 7]}),
        ]

        with patch.object(settings, "ANALYSIS_SAMPLE_ROWS", 5):
            with patch("app.services.pipeline.iter_csv_chunks", return_value=iter(chunks)) as mock_iter_chunks:
                sample_df = _load_analysis_sample("job-1")

        self.assertEqual(sample_df["value"].tolist(), [1, 2, 3, 4, 5])
        mock_iter_chunks.assert_called_once_with("job-1", chunksize=5)

    def test_load_analysis_sample_falls_back_to_empty_header_read(self):
        fallback_df = pd.DataFrame(columns=["name", "age"])

        with patch.object(settings, "ANALYSIS_SAMPLE_ROWS", 5000):
            with patch("app.services.pipeline.iter_csv_chunks", return_value=iter([])):
                with patch("app.services.pipeline.load_csv", return_value=fallback_df) as mock_load_csv:
                    sample_df = _load_analysis_sample("job-2")

        mock_load_csv.assert_called_once_with("job-2", nrows=0)
        self.assertEqual(list(sample_df.columns), ["name", "age"])
        self.assertTrue(sample_df.empty)


if __name__ == "__main__":
    unittest.main()
