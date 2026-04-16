import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.schemas.job import DataSuggestion
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.analysis.router import _resolve_is_clean_flag, router
from app.db.session import get_db
from app.schemas.job import DatasetAnalysisResponse
from app.services.ai_cleaner import analyze_dataset
from app.services.analysis_suggestions import coalesce_clean_quality_score


class _FakeSession:
    def __init__(self, job, cleaned_result=None):
        self.job = job
        self.cleaned_result = cleaned_result

    async def get(self, model, key):
        model_name = getattr(model, "__name__", "")
        if model_name == "Job":
            return self.job
        if model_name == "CleanedData":
            return self.cleaned_result
        return None

    async def commit(self):
        return None

    async def rollback(self):
        return None

    async def execute(self, *args, **kwargs):
        return None

    def add_all(self, values):
        return None


class AnalysisFlowTests(unittest.TestCase):
    def test_resolve_is_clean_flag_prefers_camel_case_when_present(self):
        self.assertTrue(_resolve_is_clean_flag(False, True))
        self.assertFalse(_resolve_is_clean_flag(True, False))
        self.assertTrue(_resolve_is_clean_flag(True, None))

    def test_clean_score_bumps_when_changes_detected_and_score_is_flat(self):
        score = coalesce_clean_quality_score(
            71,
            raw_score=71,
            previous_clean_score=None,
            changes_detected=True,
        )

        self.assertEqual(score, 72)

    def test_clean_score_does_not_bump_without_detected_changes(self):
        score = coalesce_clean_quality_score(
            71,
            raw_score=71,
            previous_clean_score=None,
            changes_detected=False,
        )

        self.assertEqual(score, 71)

    def test_analysis_route_accepts_camel_case_is_clean_query_param(self):
        job = SimpleNamespace(
            id="file-1",
            status="uploaded",
            message=None,
            quality_score=71,
            analysis=None,
        )
        cleaned_result = SimpleNamespace(
            prompt="Normalize phone values.",
            quality_score=None,
            changes_detected=True,
            analysis=None,
        )
        fake_db = _FakeSession(job=job, cleaned_result=cleaned_result)

        async def override_get_db():
            yield fake_db

        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        app.dependency_overrides[get_db] = override_get_db

        with patch(
            "app.api.routes.analysis.router.analyze_csv",
            AsyncMock(
                return_value=DatasetAnalysisResponse(
                    job_id="file-1",
                    source_type="clean",
                    quality_score=71,
                    suggestions=[],
                )
            ),
        ):
            client = TestClient(app)
            response = client.post("/api/v1/analysis/file-1?isClean=true")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source_type"], "clean")
        self.assertEqual(response.json()["quality_score"], 72)


class AnalysisMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def test_analyze_dataset_marks_llm_generated_suggestions(self):
        suggestion = DataSuggestion(
            issue_description="City values are inconsistent.",
            priority="Medium",
            resolution_prompt="Standardize city values.",
        )
        profile = {"columns": [], "dataset_issues": [], "sample_rows": []}

        with patch(
            "app.services.ai_cleaner._request_analysis_suggestions_from_llm",
            AsyncMock(return_value=[suggestion]),
        ):
            response, _ = await analyze_dataset(profile)

        self.assertTrue(response.llm_used)
        self.assertEqual(response.suggestion_source, "llm")
        self.assertEqual(len(response.suggestions), 1)

    async def test_analyze_dataset_marks_rule_based_fallback_when_llm_fails(self):
        suggestion = DataSuggestion(
            issue_description="Date formats are inconsistent.",
            priority="High",
            resolution_prompt="Normalize date values.",
        )
        profile = {"columns": [], "dataset_issues": [], "sample_rows": []}

        with patch(
            "app.services.ai_cleaner._request_analysis_suggestions_from_llm",
            AsyncMock(side_effect=TimeoutError("ollama timed out")),
        ):
            with patch(
                "app.services.ai_cleaner.generate_rule_based_suggestions",
                return_value=[suggestion],
            ):
                response, _ = await analyze_dataset(profile)

        self.assertFalse(response.llm_used)
        self.assertEqual(response.suggestion_source, "rule_based")
        self.assertEqual(len(response.suggestions), 1)


if __name__ == "__main__":
    unittest.main()
