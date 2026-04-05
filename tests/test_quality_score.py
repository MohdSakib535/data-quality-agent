import unittest

import pandas as pd

from app.services.deterministic_cleaner import build_dataset_profile, compute_quality_score


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


if __name__ == "__main__":
    unittest.main()
