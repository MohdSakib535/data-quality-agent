import unittest

from app.schemas.models import DataSuggestion
from app.services.ai_cleaner import _normalize_analysis_suggestion


class AnalysisResolutionPromptTests(unittest.TestCase):
    def test_whitespace_resolution_prompt_is_generic_and_detailed(self):
        suggestion = DataSuggestion(
            issue_description="Some text columns contain leading or trailing whitespace.",
            priority="Medium",
            resolution_prompt="Trim whitespace in name, city, and email columns.",
        )

        normalized = _normalize_analysis_suggestion(suggestion)

        self.assertIn("all affected text fields across the dataset", normalized.resolution_prompt)
        self.assertNotIn("name, city, and email", normalized.resolution_prompt)
        self.assertIn("handled consistently as blank or missing data", normalized.resolution_prompt)


if __name__ == "__main__":
    unittest.main()
