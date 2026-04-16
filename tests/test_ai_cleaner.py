import json
import unittest
from unittest.mock import patch

import pandas as pd

from app.services.ai_cleaner import (
    _extract_json_payload,
    _invoke_cleaning_batch_with_fallback,
    analyze_dataset,
    clean_dataframe_chunk,
    clean_dataset_with_prompt,
    prompt_has_deterministic_cleaning_steps,
    prompt_requires_ai_cleaning,
)


class CleanDatasetWithPromptTests(unittest.TestCase):
    def test_analyze_dataset_returns_llm_generated_suggestions(self):
        source_df = pd.DataFrame(
            [
                {"name": " Alice ", "email": None},
                {"name": "Bob", "email": "bob@example.com"},
            ]
        )
        llm_response = json.dumps(
            {
                "quality_score": 61,
                "suggestions": [
                    {
                        "issue_description": "Email values are missing in some rows.",
                        "priority": "High",
                        "resolution_prompt": "Fill missing email values with N/A and trim name whitespace.",
                    }
                ],
            }
        )

        class FakeChain:
            def __init__(self, response):
                self.response = response

            def __or__(self, other):
                return self

            def invoke(self, payload):
                return self.response

        class FakePrompt:
            def __init__(self, response):
                self.response = response

            def __or__(self, other):
                return FakeChain(self.response)

        with patch("app.services.ai_cleaner.analysis_prompt_template", FakePrompt(llm_response)):
            with patch("app.services.ai_cleaner.ChatOllama", return_value=object()):
                response = analyze_dataset(source_df)

        self.assertEqual(response.quality_score, 61)
        self.assertEqual(len(response.suggestions), 1)
        self.assertEqual(response.suggestions[0].issue_description, "Email values are missing in some rows.")
        self.assertEqual(
            response.suggestions[0].resolution_prompt,
            "Fill missing email values with N/A and trim name whitespace.",
        )

    def test_analyze_dataset_raises_when_llm_analysis_fails(self):
        source_df = pd.DataFrame([{"name": "Alice"}])

        with patch("app.services.ai_cleaner.ChatOllama", side_effect=RuntimeError("ollama offline")):
            with self.assertRaises(RuntimeError) as exc:
                analyze_dataset(source_df)

        self.assertIn("LLM analysis failed", str(exc.exception))

    def test_extract_json_payload_accepts_python_literal_style_response(self):
        raw = """```json
        [{'name': 'Alice', 'active': True, 'notes': None}]
        ```"""
        payload = _extract_json_payload(raw)
        self.assertEqual(payload, [{"name": "Alice", "active": True, "notes": None}])

    def test_invoke_cleaning_batch_with_fallback_splits_failed_batch(self):
        rows = [{"city": "mumbai"}, {"city": "delhi"}]

        def flaky_invoke(chain, batch_json, user_prompt, expected_rows):
            if expected_rows > 1:
                raise ValueError("non-json response")
            single = json.loads(batch_json)[0]
            return [{"city": single["city"].title()}]

        with patch("app.services.ai_cleaner._invoke_cleaning_batch", side_effect=flaky_invoke):
            cleaned = _invoke_cleaning_batch_with_fallback(
                chain="fake-chain",
                batch_records=rows,
                user_prompt="Standardize capitalization for city values.",
            )

        self.assertEqual(cleaned, [{"city": "Mumbai"}, {"city": "Delhi"}])

    def test_invoke_cleaning_batch_with_fallback_keeps_rows_when_model_keeps_mismatching(self):
        rows = [{"city": "mumbai"}, {"city": "delhi"}]

        with patch(
            "app.services.ai_cleaner._invoke_cleaning_batch",
            side_effect=ValueError("Batch size mismatch. Expected 2, got 6"),
        ):
            cleaned = _invoke_cleaning_batch_with_fallback(
                chain="fake-chain",
                batch_records=rows,
                user_prompt="Standardize capitalization for city values.",
            )

        self.assertEqual(cleaned, rows)

    def test_trims_string_cells_when_prompt_requests_whitespace_cleanup(self):
        source_df = pd.DataFrame(
            [
                {"name": "  Alice  ", "title": " Engineer ", "score": 10},
                {"name": "\tBob\n", "title": "   ", "score": 20},
                {"name": None, "title": "Manager", "score": 30},
            ]
        )

        cleaned_df = clean_dataset_with_prompt(
            source_df,
            "Trim leading or trailing whitespace in all affected text fields.",
        )

        self.assertEqual(
            cleaned_df.to_dict(orient="records"),
            [
                {"name": "Alice", "title": "Engineer", "score": 10},
                {"name": "Bob", "title": "", "score": 20},
                {"name": None, "title": "Manager", "score": 30},
            ],
        )

    def test_duplicate_only_prompt_preserves_non_requested_fields(self):
        source_df = pd.DataFrame(
            [
                {
                    " Full Name ": "John Doe",
                    "Email Address": "john.doe@example.com",
                    "Phone Number": "9876543210",
                    "City": "Delhi",
                    "Join Date": "01/02/2025",
                    "Amount / Score": "  ₹12,500  ",
                    "Status": "Paid",
                    "Job Title": "ui ux desginer",
                },
                {
                    " Full Name ": "John Doe",
                    "Email Address": "john.doe@example.com",
                    "Phone Number": "9876543210",
                    "City": "Delhi",
                    "Join Date": "01/02/2025",
                    "Amount / Score": "  ₹12,500  ",
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
        cleaned_df = clean_dataset_with_prompt(source_df, prompt)

        self.assertEqual(len(cleaned_df), 1)
        self.assertEqual(cleaned_df.iloc[0]["Amount / Score"], "  ₹12,500  ")

    def test_ai_cleaning_updates_only_prompt_target_columns(self):
        source_df = pd.DataFrame(
            [
                {"name": "Alice", "job_title": "ui ux desginer"},
                {"name": "Bob", "job_title": "Sr. Engneer"},
            ]
        )
        seen_batch_rows = []

        def fake_invoke_cleaning_batch(chain, batch_json, user_prompt, expected_rows):
            rows = json.loads(batch_json)
            seen_batch_rows.extend(rows)
            updated_rows = []
            for row in rows:
                self.assertNotIn("name", row)
                updated_rows.append(
                    {
                        "name": "SHOULD_NOT_BE_APPLIED",
                        "job_title": row["job_title"].replace("desginer", "designer").replace("Engneer", "Engineer"),
                    }
                )
            return updated_rows

        with patch("app.services.ai_cleaner._invoke_cleaning_batch", side_effect=fake_invoke_cleaning_batch):
            cleaned_df = clean_dataframe_chunk(
                source_df,
                "Standardize spelling in job_title column.",
                chain="fake-chain",
            )

        self.assertTrue(all(set(row.keys()) == {"job_title"} for row in seen_batch_rows))
        self.assertEqual(cleaned_df["name"].tolist(), ["Alice", "Bob"])
        self.assertEqual(cleaned_df["job_title"].tolist(), ["ui ux designer", "Sr. Engineer"])

    def test_mixed_whitespace_and_ai_prompt_runs_both_cleaning_paths(self):
        source_df = pd.DataFrame(
            [
                {"name": " Alice ", "job_title": "ui ux desginer"},
                {"name": " Bob ", "job_title": "Sr. Engneer"},
            ]
        )
        seen_batch_rows = []

        def fake_invoke_cleaning_batch(chain, batch_json, user_prompt, expected_rows):
            rows = json.loads(batch_json)
            seen_batch_rows.extend(rows)
            return [
                {
                    "job_title": row["job_title"].replace("desginer", "designer").replace("Engneer", "Engineer"),
                }
                for row in rows
            ]

        with patch("app.services.ai_cleaner._invoke_cleaning_batch", side_effect=fake_invoke_cleaning_batch):
            cleaned_df = clean_dataframe_chunk(
                source_df,
                "Trim whitespace in all text fields and standardize spelling in job_title column.",
                chain="fake-chain",
            )

        self.assertTrue(all(set(row.keys()) == {"job_title"} for row in seen_batch_rows))
        self.assertEqual(cleaned_df["name"].tolist(), ["Alice", "Bob"])
        self.assertEqual(cleaned_df["job_title"].tolist(), ["ui ux designer", "Sr. Engineer"])

    def test_ai_cleaning_reuses_duplicate_rows_within_chunk(self):
        source_df = pd.DataFrame(
            [
                {"city": "mumbai"},
                {"city": "mumbai"},
                {"city": "delhi"},
            ]
        )
        expected_batch_sizes = []

        def fake_invoke_cleaning_batch(chain, batch_json, user_prompt, expected_rows):
            expected_batch_sizes.append(expected_rows)
            rows = json.loads(batch_json)
            return [{"city": row["city"].title()} for row in rows]

        with patch("app.services.ai_cleaner._invoke_cleaning_batch", side_effect=fake_invoke_cleaning_batch):
            cleaned_df = clean_dataframe_chunk(
                source_df,
                "Standardize capitalization for city values.",
                chain="fake-chain",
            )

        self.assertEqual(expected_batch_sizes, [2])
        self.assertEqual(cleaned_df["city"].tolist(), ["Mumbai", "Mumbai", "Delhi"])

    def test_analyzer_generated_phone_prompt_stays_deterministic(self):
        source_df = pd.DataFrame(
            [
                {"Phone Number": "(987) 654-3210", "name": "Alice"},
                {"Phone Number": "+1 202-555-0189", "name": "Bob"},
            ]
        )
        prompt = (
            "Only in column(s) 'Phone Number', standardize phone values into one consistent phone format, "
            "remove punctuation noise, preserve country codes when present, keep valid phone-like values, "
            "and leave unrelated columns untouched."
        )

        self.assertFalse(prompt_requires_ai_cleaning(prompt))
        self.assertTrue(prompt_has_deterministic_cleaning_steps(prompt))

        cleaned_df = clean_dataframe_chunk(source_df, prompt)

        self.assertEqual(cleaned_df["Phone Number"].tolist(), ["9876543210", "+12025550189"])
        self.assertEqual(cleaned_df["name"].tolist(), ["Alice", "Bob"])


if __name__ == "__main__":
    unittest.main()
