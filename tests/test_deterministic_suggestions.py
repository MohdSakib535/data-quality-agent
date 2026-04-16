import unittest

from app.core.config import settings
from app.services.deterministic_cleaner import generate_rule_based_suggestions


class DeterministicSuggestionTests(unittest.TestCase):
    def test_rule_based_suggestions_include_explicit_target_columns(self):
        profile = {
            "columns": [
                {
                    "name": "Join Date",
                    "invalid_date_percent": 42.0,
                },
                {
                    "name": "Amount / Score",
                    "invalid_numeric_percent": 17.0,
                },
                {
                    "name": "City",
                    "casing_inconsistency_percent": 25.0,
                    "whitespace_inconsistency_percent": 10.0,
                },
            ],
            "duplicate_row_percent": 0.0,
        }

        suggestions = generate_rule_based_suggestions(profile)
        prompts = [suggestion.resolution_prompt for suggestion in suggestions]

        self.assertTrue(any("'Join Date'" in prompt for prompt in prompts))
        self.assertTrue(any(settings.DATE_OUTPUT_FORMAT in prompt for prompt in prompts))
        self.assertTrue(any("'Amount / Score'" in prompt for prompt in prompts))
        self.assertTrue(any("'City'" in prompt for prompt in prompts))

    def test_missing_value_suggestion_uses_configured_placeholder(self):
        profile = {
            "columns": [
                {
                    "name": "Email",
                    "null_percent": 33.0,
                }
            ],
            "duplicate_row_percent": 0.0,
        }

        suggestions = generate_rule_based_suggestions(profile)

        self.assertEqual(len(suggestions), 1)
        self.assertIn(settings.NULL_OUTPUT_TOKEN, suggestions[0].resolution_prompt)
        self.assertIn("'Email'", suggestions[0].resolution_prompt)

    def test_consistency_only_format_issues_still_generate_cleanup_suggestions(self):
        profile = {
            "columns": [
                {
                    "name": "Phone Number",
                    "phone_format_inconsistency_percent": 66.0,
                },
                {
                    "name": "Join Date",
                    "date_format_inconsistency_percent": 50.0,
                },
                {
                    "name": "Amount",
                    "numeric_format_inconsistency_percent": 33.0,
                },
            ],
            "duplicate_row_percent": 0.0,
        }

        suggestions = generate_rule_based_suggestions(profile)
        prompts = [suggestion.resolution_prompt for suggestion in suggestions]

        self.assertTrue(any("'Phone Number'" in prompt for prompt in prompts))
        self.assertTrue(any("'Join Date'" in prompt for prompt in prompts))
        self.assertTrue(any("'Amount'" in prompt for prompt in prompts))


if __name__ == "__main__":
    unittest.main()
