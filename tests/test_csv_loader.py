import io
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from app.core.config import settings
from app.services.csv_loader import (
    is_supported_upload_file,
    load_csv,
    save_upload_file,
)


class CsvLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.upload_dir_patch = patch.object(settings, "UPLOAD_DIR", self.temp_dir.name)
        self.upload_dir_patch.start()

    def tearDown(self):
        self.upload_dir_patch.stop()
        self.temp_dir.cleanup()

    def test_supported_extension_validation(self):
        self.assertTrue(is_supported_upload_file(SimpleNamespace(filename="sample.csv")))
        self.assertTrue(is_supported_upload_file(SimpleNamespace(filename="sample.CSV")))
        self.assertTrue(is_supported_upload_file(SimpleNamespace(filename="sample.xlsx")))
        self.assertTrue(is_supported_upload_file(SimpleNamespace(filename="sample.xls")))
        self.assertFalse(is_supported_upload_file(SimpleNamespace(filename="sample.txt")))
        self.assertFalse(is_supported_upload_file(SimpleNamespace(filename=None)))

    def test_save_upload_file_saves_csv_without_conversion(self):
        upload = SimpleNamespace(
            filename="customers.csv",
            file=io.BytesIO(b"name,age\nAlice,30\nBob,27\n"),
        )
        job_id = save_upload_file(upload)

        saved_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}.csv")
        self.assertTrue(os.path.exists(saved_path))

        saved_df = load_csv(job_id)
        self.assertEqual(
            saved_df.to_dict(orient="records"),
            [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 27}],
        )

    @patch("app.services.csv_loader.pd.read_excel")
    def test_save_upload_file_converts_excel_to_csv(self, mock_read_excel):
        mock_read_excel.return_value = pd.DataFrame(
            [{"name": "Alice", "department": "Ops"}, {"name": "Bob", "department": "Finance"}]
        )
        upload = SimpleNamespace(filename="employees.xlsx", file=io.BytesIO(b"fake_excel_bytes"))

        job_id = save_upload_file(upload)
        saved_df = load_csv(job_id)

        self.assertEqual(mock_read_excel.call_count, 1)
        self.assertEqual(
            saved_df.to_dict(orient="records"),
            [{"name": "Alice", "department": "Ops"}, {"name": "Bob", "department": "Finance"}],
        )

    def test_save_upload_file_rejects_unsupported_type(self):
        upload = SimpleNamespace(filename="employees.json", file=io.BytesIO(b"{}"))

        with self.assertRaisesRegex(
            ValueError, "Only CSV or Excel files \\(\\.csv, \\.xlsx, \\.xls\\) are allowed\\."
        ):
            save_upload_file(upload)


if __name__ == "__main__":
    unittest.main()
