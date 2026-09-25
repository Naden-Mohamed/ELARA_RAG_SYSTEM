import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from src.services.data_service import DocumentParserService


@pytest.fixture
def service():
    return DocumentParserService()


class TestValidateUploadedFile:
    def test_rejects_disallowed_content_type(self, service):
        fake_file = MagicMock(content_type="application/x-msdownload", size=1000)
        is_valid, _ = service.validate_uploaded_file(file=fake_file)
        assert is_valid is False

    def test_accepts_allowed_pdf(self, service):
        fake_file = MagicMock(content_type="application/pdf", size=1000)
        is_valid, _ = service.validate_uploaded_file(file=fake_file)
        assert is_valid is True

    def test_rejects_oversized_file(self, service):
        too_big = service.settings.FILE_MAX_SIZE_MB * 1024 * 1024 + 1
        fake_file = MagicMock(content_type="application/pdf", size=too_big)
        is_valid, _ = service.validate_uploaded_file(file=fake_file)
        assert is_valid is False

    def test_rejects_file_with_missing_size(self, service):
        fake_file = MagicMock(content_type="application/pdf", size=None)
        is_valid, _ = service.validate_uploaded_file(file=fake_file)
        assert is_valid is False


class TestFilenameSanitization:
    def test_generated_filename_has_random_prefix(self, service):
        name = service.generate_unique_filename("report.pdf")
        assert name.endswith("_report.pdf")
        prefix = name.split("_")[0]
        assert len(prefix) == 5

    @pytest.mark.parametrize(
        "malicious_name",
        [
            "../../etc/passwd",
            "../../../secrets.env",
            "..\\..\\windows\\system32\\config",
            "/etc/passwd",
        ],
    )
    def test_path_traversal_is_stripped(self, service, malicious_name):
        result = service.generate_unique_filename(malicious_name)
        assert "/" not in result
        assert "\\" not in result
        assert ".." not in result

    def test_none_filename_does_not_crash(self, service):
        result = service.generate_unique_filename(None)
        assert result
