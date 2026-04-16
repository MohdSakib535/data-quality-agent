import unittest

from app.services.analysis_suggestions import (
    build_data_suggestion,
    coalesce_clean_quality_score,
)
from app.schemas.job import DataSuggestion


class CleanAnalysisQualityScoreTests(unittest.TestCase):
    def test_clean_score_does_not_drop_below_raw_score(self):
        score = coalesce_clean_quality_score(
            68,
            raw_score=81,
            previous_clean_score=None,
        )

        self.assertEqual(score, 81)

    def test_clean_score_does_not_drop_below_previous_clean_score(self):
        score = coalesce_clean_quality_score(
            72,
            raw_score=70,
            previous_clean_score=88,
        )

        self.assertEqual(score, 88)

    def test_clean_score_keeps_current_value_when_it_is_highest(self):
        score = coalesce_clean_quality_score(
            91,
            raw_score=84,
            previous_clean_score=89,
        )

        self.assertEqual(score, 91)

    def test_build_data_suggestion_extracts_target_columns_and_prompt_type(self):
        suggestion = DataSuggestion(
            issue_description="Date-like column(s) 'Join Date' contain inconsistent or invalid date values.",
            priority="High",
            resolution_prompt=(
                "Only in column(s) 'Join Date', convert parseable date values to %Y-%m-%d, "
                "preserve already valid dates, and do not modify non-date columns."
            ),
        )

        payload = build_data_suggestion(suggestion, suggestion_id="s-1")

        self.assertEqual(payload.id, "s-1")
        self.assertEqual(payload.cleaning_prompt_type, "date_normalization")
        self.assertEqual(payload.target_columns, ["Join Date"])

    def test_build_data_suggestion_defaults_to_generic_cleaning_when_category_is_unknown(self):
        suggestion = DataSuggestion(
            issue_description="Some rows contain inconsistent labels.",
            priority="Medium",
            resolution_prompt="Review the affected values and standardize them carefully without changing unrelated columns.",
        )

        payload = build_data_suggestion(suggestion, suggestion_id="s-2")

        self.assertEqual(payload.cleaning_prompt_type, "generic_cleaning")
        self.assertEqual(payload.target_columns, [])


if __name__ == "__main__":
    unittest.main()
