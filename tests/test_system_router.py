import unittest
import importlib.util
from pathlib import Path

import httpx


def _load_system_router_module():
    module_path = Path(__file__).resolve().parents[1] / "app" / "api" / "routes" / "system" / "router.py"
    spec = importlib.util.spec_from_file_location("test_system_router_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_system_router = _load_system_router_module()
_describe_generation_error = _system_router._describe_generation_error


class SystemRouterTests(unittest.TestCase):
    def test_describe_generation_error_includes_timeout_details(self):
        request = httpx.Request("POST", "http://host.docker.internal:11434/api/generate")
        exc = httpx.ReadTimeout("", request=request)

        payload = _describe_generation_error(exc)

        self.assertEqual(payload["model_error_type"], "ReadTimeout")
        self.assertIn("timed out", payload["model_error"])
        self.assertIsNone(payload["model_error_status_code"])
        self.assertIsNone(payload["model_error_response"])

    def test_describe_generation_error_includes_http_status_and_response_excerpt(self):
        request = httpx.Request("POST", "http://host.docker.internal:11434/api/generate")
        response = httpx.Response(
            500,
            request=request,
            text='{"error":"model runner crashed"}',
        )
        exc = httpx.HTTPStatusError("server error", request=request, response=response)

        payload = _describe_generation_error(exc)

        self.assertEqual(payload["model_error_type"], "HTTPStatusError")
        self.assertEqual(payload["model_error_status_code"], 500)
        self.assertIn("HTTP 500", payload["model_error"])
        self.assertEqual(payload["model_error_response"], '{"error":"model runner crashed"}')

    def test_describe_generation_error_falls_back_to_repr_when_message_is_empty(self):
        exc = RuntimeError("")

        payload = _describe_generation_error(exc)

        self.assertEqual(payload["model_error_type"], "RuntimeError")
        self.assertEqual(payload["model_error"], "RuntimeError: RuntimeError('')")


if __name__ == "__main__":
    unittest.main()
