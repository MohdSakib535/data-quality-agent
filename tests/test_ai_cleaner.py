import json
import unittest
from unittest.mock import patch

import pandas as pd

from app.services.ai_cleaner import clean_dataset_with_prompt


class CleanDatasetWithPromptTests(unittest.TestCase):
    def test_trims_string_cells_before_prompt_cleaning(self):
        source_df = pd.DataFrame(
            [
                {"name": "  Alice  ", "title": " Engineer ", "score": 10},
                {"name": "\tBob\n", "title": "   ", "score": 20},
                {"name": None, "title": "Manager", "score": 30},
            ]
        )
        captured_batches = []

        def fake_invoke_cleaning_batch(chain, batch_json, user_prompt, expected_rows):
            self.assertEqual(chain, "fake-chain")
            self.assertEqual(user_prompt, "normalize titles")
            self.assertEqual(expected_rows, 3)

            rows = json.loads(batch_json)
            captured_batches.append(rows)
            return rows

        with patch("app.services.ai_cleaner._build_cleaning_chain", return_value="fake-chain"):
            with patch(
                "app.services.ai_cleaner._invoke_cleaning_batch",
                side_effect=fake_invoke_cleaning_batch,
            ):
                cleaned_df = clean_dataset_with_prompt(source_df, "normalize titles")

        self.assertEqual(len(captured_batches), 1)
        self.assertEqual(
            captured_batches[0],
            [
                {"name": "Alice", "title": "Engineer", "score": 10},
                {"name": "Bob", "title": "", "score": 20},
                {"name": None, "title": "Manager", "score": 30},
            ],
        )
        self.assertEqual(cleaned_df.to_dict(orient="records"), captured_batches[0])

    def test_removes_exact_duplicate_rows_before_prompt_cleaning(self):
        source_df = pd.DataFrame(
            [
                {
                    " Full Name ": "John Doe",
                    "Email Address": "john.doe@example.com",
                    "Phone Number": "9876543210",
                    "City": "Delhi",
                    "Join Date": "01/02/2025",
                    "Amount / Score": "₹12,500",
                    "Status": "Paid",
                    "Job Title": "ui ux desginer",
                },
                {
                    " Full Name ": "John Doe",
                    "Email Address": "john.doe@example.com",
                    "Phone Number": "9876543210",
                    "City": "Delhi",
                    "Join Date": "01/02/2025",
                    "Amount / Score": "₹12,500",
                    "Status": "Paid",
                    "Job Title": "ui ux desginer",
                },
            ]
        )
        prompt = (
            "Identify exact duplicate records across the dataset and remove redundant copies while "
            "keeping one canonical version of each repeated row. Preserve legitimate repeated values "
            "that are not true duplicates, and avoid merging rows unless all corresponding fields "
            "represent the same record."
        )
        captured_batches = []

        def fake_invoke_cleaning_batch(chain, batch_json, user_prompt, expected_rows):
            self.assertEqual(chain, "fake-chain")
            self.assertEqual(user_prompt, prompt)
            self.assertEqual(expected_rows, 1)

            rows = json.loads(batch_json)
            captured_batches.append(rows)
            return rows

        with patch("app.services.ai_cleaner._build_cleaning_chain", return_value="fake-chain"):
            with patch(
                "app.services.ai_cleaner._invoke_cleaning_batch",
                side_effect=fake_invoke_cleaning_batch,
            ):
                cleaned_df = clean_dataset_with_prompt(source_df, prompt)

        self.assertEqual(
            captured_batches[0],
            [
                {
                    " Full Name ": "John Doe",
                    "Email Address": "john.doe@example.com",
                    "Phone Number": "9876543210",
                    "City": "Delhi",
                    "Join Date": "01/02/2025",
                    "Amount / Score": "₹12,500",
                    "Status": "Paid",
                    "Job Title": "ui ux desginer",
                }
            ],
        )
        self.assertEqual(len(cleaned_df), 1)


if __name__ == "__main__":
    unittest.main()
