import io

import pytest


def _pdf_bytes() -> bytes:
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _patch_upload_dir(monkeypatch, tmp_path):

    import src.services.data_service as data_service_module

    original_init = data_service_module.DocumentParserService.__init__

    def patched_init(self):
        original_init(self)
        self.files_path = str(tmp_path)

    monkeypatch.setattr(
        data_service_module.DocumentParserService, "__init__", patched_init
    )


class TestUploadRequiresAuth:
    """
    Regression test for the fix that added Depends(get_current_user) to
    /data/upload. If this test fails with a 200, the endpoint is unprotected.
    """

    @pytest.mark.asyncio
    async def test_upload_without_token_is_rejected(self, client):
        files = {"file": ("doc.pdf", _pdf_bytes(), "application/pdf")}
        resp = await client.post("/data/upload", files=files)
        assert resp.status_code in (401, 403)


class TestUpload:
    @pytest.mark.asyncio
    async def test_upload_valid_pdf_succeeds(
        self, client, registered_user, tmp_path, monkeypatch
    ):
        headers, _, _ = registered_user
        _patch_upload_dir(monkeypatch, tmp_path)

        files = {"file": ("resume.pdf", _pdf_bytes(), "application/pdf")}
        resp = await client.post("/data/upload", headers=headers, files=files)
        body = resp.json()
        assert body["status"] == "success" or "FILE_UPLOADED" in str(
            body.get("status", "")
        )
        assert "document_id" in body["data"]

    @pytest.mark.asyncio
    async def test_upload_rejects_disallowed_type(self, client, registered_user):
        headers, _, _ = registered_user
        files = {"file": ("virus.exe", b"MZ\x90\x00", "application/x-msdownload")}
        resp = await client.post("/data/upload", headers=headers, files=files)
        body = resp.json()
        assert body["status_code"] == 400


class TestPathTraversalRegression:
    """
    Regression test for the path-traversal fix: an uploaded filename
    containing directory-escape sequences must never be written outside
    the intended upload directory.
    """

    @pytest.mark.asyncio
    async def test_malicious_filename_does_not_escape_upload_dir(
        self, client, registered_user, tmp_path, monkeypatch
    ):
        headers, _, _ = registered_user
        _patch_upload_dir(monkeypatch, tmp_path)

        files = {"file": ("../../../evil.pdf", _pdf_bytes(), "application/pdf")}
        resp = await client.post("/data/upload", headers=headers, files=files)
        assert resp.status_code in (200, 201) or resp.json()["status_code"] in (
            200,
            201,
        )

        # Nothing should have been written above tmp_path
        escaped_path = tmp_path.parent.parent.parent / "evil.pdf"
        assert not escaped_path.exists()
        # Confirm the file landed inside the sandboxed directory instead
        written_files = list(tmp_path.iterdir())
        assert len(written_files) == 1
        assert ".." not in written_files[0].name
        assert "/" not in written_files[0].name


class TestIngestRequiresValidDocument:
    @pytest.mark.asyncio
    async def test_ingest_unknown_document_id_returns_404(
        self, client, registered_user
    ):
        headers, _, _ = registered_user
        resp = await client.post(
            "/data/ingest",
            headers=headers,
            params={"document_id": "64b64b64b64b64b64b64b64"},
        )
        body = resp.json()
        assert body["status_code"] == 404
