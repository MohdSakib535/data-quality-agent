import unittest

from app.models.semantic_column_metadata import SemanticColumnMetadata
from app.services.chat_service import build_prompt, build_query_plan_prompt
from app.services.semantic_layer import (
    _build_synonyms,
    _score_metadata_candidate,
    format_semantic_context_for_prompt,
)


class ChatSemanticPlannerTests(unittest.TestCase):
    def test_build_prompt_includes_semantic_hints_and_plan(self):
        prompt = build_prompt(
            question="Show top clients by revenue",
            schema=[{"column_name": "cleaned_data", "data_type": "json"}],
            table_name="cleaned_data",
            row_schema=[{"column_name": "customer_name", "data_type": "text"}],
            semantic_context=[
                {
                    "column_name": "customer_name",
                    "data_type": "text",
                    "source_type": "json_row",
                    "synonyms": ["customer", "client"],
                    "sample_values": ["Acme Corp"],
                    "score": 8.5,
                }
            ],
            query_plan={"intent": "top clients by revenue", "confidence": 0.81},
        )

        self.assertIn("Top semantic hints for this question", prompt)
        self.assertIn("customer_name", prompt)
        self.assertIn("Structured query plan from a previous planning step", prompt)
        self.assertIn('"intent": "top clients by revenue"', prompt)

    def test_build_query_plan_prompt_requires_json_shape(self):
        prompt = build_query_plan_prompt(
            question="Monthly revenue trend",
            schema=[{"column_name": "job_id", "data_type": "text"}],
            table_name="cleaned_data",
            row_schema=[{"column_name": "amount", "data_type": "numeric"}],
            semantic_context=[],
        )

        self.assertIn("Return ONLY valid JSON", prompt)
        self.assertIn('"required_columns"', prompt)
        self.assertIn("needs_clarification", prompt)

    def test_synonym_builder_is_header_and_sample_driven(self):
        synonyms = _build_synonyms(
            "contact_field",
            data_type="text",
            sample_values=["alice@example.com", "bob@example.com"],
        )
        lowered = {value.lower() for value in synonyms}
        self.assertIn("contact field", lowered)
        self.assertIn("email", lowered)
        self.assertIn("email address", lowered)

    def test_scoring_uses_synonyms_and_sample_values(self):
        metadata = SemanticColumnMetadata(
            job_id="job-1",
            table_name="cleaned_data",
            source_type="json_row",
            column_name="cust_id",
            data_type="text",
            description="Customer identifier",
            synonyms=["customer", "client", "customer id"],
            sample_values=["ACME", "Globex"],
        )
        score = _score_metadata_candidate(
            question="Which client is highest revenue for ACME?",
            question_tokens={"which", "client", "highest", "revenue", "acme"},
            metadata=metadata,
        )
        self.assertGreater(score, 0)

    def test_semantic_context_formatter(self):
        text = format_semantic_context_for_prompt(
            [
                {
                    "column_name": "order_date",
                    "data_type": "date",
                    "source_type": "json_row",
                    "synonyms": ["order date", "purchase date"],
                    "sample_values": ["2026-01-05"],
                    "score": 7.2,
                }
            ]
        )
        self.assertIn("order_date", text)
        self.assertIn("purchase date", text)


if __name__ == "__main__":
    unittest.main()
