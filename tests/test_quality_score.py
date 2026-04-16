import unittest

import pandas as pd

from app.services.deterministic_cleaner import build_dataset_profile, compute_quality_score
from app.services.ai_cleaner import clean_dataset_with_prompt


class QualityScoreTests(unittest.TestCase):
    def test_messy_dataset_scores_significantly_lower(self):
        df = pd.DataFrame(
            [
                {
                    "customer_id": "A001",
                    "email": "bad-email",
                    "phone": "123",
                    "amount": "Rs ???",
                    "city": " Delhi ",
                    "signup_date": "32/13/2025",
                },
                {
                    "customer_id": "A001",
                    "email": "",
                    "phone": "abcd",
                    "amount": "$12x",
                    "city": "delhi",
                    "signup_date": "not-a-date",
                },
                {
                    "customer_id": "A001",
                    "email": None,
                    "phone": None,
                    "amount": "",
                    "city": "DELHI",
                    "signup_date": "",
                },
            ]
        )

        profile = build_dataset_profile(df)
        score = compute_quality_score(profile)

        self.assertLess(score, 60)

    def test_clean_dataset_scores_high(self):
        df = pd.DataFrame(
            [
                {
                    "customer_id": "A001",
                    "email": "alice@example.com",
                    "phone": "919876543210",
                    "amount": "12500",
                    "city": "Delhi",
                    "signup_date": "2025-01-02",
                },
                {
                    "customer_id": "A002",
                    "email": "bob@example.com",
                    "phone": "919812345678",
                    "amount": "9800",
                    "city": "Mumbai",
                    "signup_date": "2025-01-03",
                },
            ]
        )

        profile = build_dataset_profile(df)
        score = compute_quality_score(profile)

        self.assertGreaterEqual(score, 95)

    def test_normalization_prompt_improves_quality_score_for_consistent_formatting(self):
        raw_df = pd.DataFrame(
            [
                {
                    "customer_id": "A001",
                    "phone": "(987) 654-3210",
                    "signup_date": "01/02/2025",
                    "amount": "$12,500.00",
                },
                {
                    "customer_id": "A002",
                    "phone": "9876543211",
                    "signup_date": "2025-01-03",
                    "amount": "12500",
                },
                {
                    "customer_id": "A003",
                    "phone": "+1 202-555-0189",
                    "signup_date": "3 Jan 2025",
                    "amount": "12 500",
                },
            ]
        )

        raw_score = compute_quality_score(build_dataset_profile(raw_df))

        cleaned_df = clean_dataset_with_prompt(
            raw_df,
            "Only in column(s) 'phone', 'signup_date', 'amount', standardize phone values into one consistent phone format, "
            "convert parseable date values to YYYY-MM-DD, and remove numeric formatting noise such as currency symbols and separators.",
        )
        cleaned_score = compute_quality_score(build_dataset_profile(cleaned_df))

        self.assertGreater(cleaned_score, raw_score)


if __name__ == "__main__":
    unittest.main()
