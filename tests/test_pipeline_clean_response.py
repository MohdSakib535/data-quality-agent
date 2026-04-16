import unittest
from unittest.mock import patch

from app.services.pipeline import clean_csv_with_prompt


class CleanCsvWithPromptResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_clean_response_includes_execution_metadata(self):
        with patch(
            "app.services.pipeline._clean_csv_file_with_prompt",
            return_value=(
                "s3://bucket/job-1.csv",
                [{"name": "Alice"}],
                10,
                "value_mapping_ai",
                ["name"],
                True,
            ),
        ):
            response = await clean_csv_with_prompt(
                "job-1",
                "Standardize spelling in name column.",
                use_cleaned_source=False,
            )

        self.assertEqual(response.prompt, "Standardize spelling in name column.")
        self.assertEqual(response.source_type, "raw")
        self.assertEqual(response.cleaning_strategy, "value_mapping_ai")
        self.assertTrue(response.llm_used)
        self.assertEqual(response.target_columns, ["name"])
        self.assertEqual(response.preview_rows_returned, 1)
        self.assertTrue(response.preview_limited)
        self.assertTrue(response.changes_detected)
        self.assertIn("Changes were detected.", response.message)
        self.assertIn("Target columns: name.", response.message)

    async def test_clean_response_reports_raw_fallback_when_cleaned_source_is_missing(self):
        with patch(
            "app.services.pipeline._clean_csv_file_with_prompt",
            side_effect=[
                FileNotFoundError("missing cleaned source"),
                (
                    "s3://bucket/job-2.csv",
                    [],
                    0,
                    "deterministic",
                    [],
                    False,
                ),
            ],
        ):
            response = await clean_csv_with_prompt(
                "job-2",
                "Normalize headers to snake_case.",
                use_cleaned_source=True,
            )

        self.assertEqual(response.source_type, "raw")
        self.assertEqual(response.cleaning_strategy, "deterministic")
        self.assertFalse(response.llm_used)
        self.assertFalse(response.preview_limited)
        self.assertFalse(response.changes_detected)
        self.assertIn("original upload", response.message)

    async def test_clean_response_reports_privacy_safe_deterministic_fallback(self):
        with patch(
            "app.services.pipeline._clean_csv_file_with_prompt",
            return_value=(
                "s3://bucket/job-3.csv",
                [{"Phone Number": "9876543210"}],
                2,
                "deterministic_privacy_fallback",
                ["Phone Number"],
                True,
            ),
        ):
            response = await clean_csv_with_prompt(
                "job-3",
                "Only in column(s) 'Phone Number', standardize phone values into one consistent phone format.",
                use_cleaned_source=False,
            )

        self.assertEqual(response.cleaning_strategy, "deterministic_privacy_fallback")
        self.assertFalse(response.llm_used)
        self.assertTrue(response.changes_detected)
        self.assertIn("privacy-safe fallback", response.message)
        self.assertIn("privacy-sensitive columns", response.message)


if __name__ == "__main__":
    unittest.main()
