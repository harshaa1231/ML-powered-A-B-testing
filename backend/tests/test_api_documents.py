from tests.conftest import requires_db


async def _auth_headers(client, email: str) -> dict[str, str]:
    resp = await client.post("/api/auth/signup", json={"email": email, "password": "hunter2pass", "persona": "business"})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@requires_db
async def test_upload_list_and_delete_document(client) -> None:
    headers = await _auth_headers(client, "doc-uploader@example.com")

    uploaded = await client.post(
        "/api/documents/upload",
        headers=headers,
        files={"file": ("notes.txt", b"Our checkout redesign shipped last quarter.", "text/plain")},
    )
    assert uploaded.status_code == 201
    doc_id = uploaded.json()["id"]
    assert uploaded.json()["filename"] == "notes.txt"
    assert uploaded.json()["file_type"] == "txt"

    listed = await client.get("/api/documents", headers=headers)
    assert listed.status_code == 200
    assert len(listed.json()) == 1

    fetched = await client.get(f"/api/documents/{doc_id}", headers=headers)
    assert fetched.status_code == 200
    assert "checkout redesign" in fetched.json()["content"]

    deleted = await client.delete(f"/api/documents/{doc_id}", headers=headers)
    assert deleted.status_code == 204

    listed_after = await client.get("/api/documents", headers=headers)
    assert listed_after.json() == []


@requires_db
async def test_upload_rejects_unsupported_file_type(client) -> None:
    headers = await _auth_headers(client, "doc-bad-type@example.com")
    resp = await client.post(
        "/api/documents/upload",
        headers=headers,
        files={"file": ("photo.png", b"fake image bytes", "image/png")},
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


@requires_db
async def test_documents_are_scoped_per_user(client) -> None:
    headers_a = await _auth_headers(client, "doc-user-a@example.com")
    headers_b = await _auth_headers(client, "doc-user-b@example.com")

    uploaded = await client.post(
        "/api/documents/upload",
        headers=headers_a,
        files={"file": ("private.txt", b"User A's private notes.", "text/plain")},
    )
    doc_id = uploaded.json()["id"]

    listed_b = await client.get("/api/documents", headers=headers_b)
    assert listed_b.json() == []

    fetched_by_b = await client.get(f"/api/documents/{doc_id}", headers=headers_b)
    assert fetched_by_b.status_code == 404

    deleted_by_b = await client.delete(f"/api/documents/{doc_id}", headers=headers_b)
    assert deleted_by_b.status_code == 404


@requires_db
async def test_documents_require_auth(client) -> None:
    resp = await client.get("/api/documents")
    assert resp.status_code == 401


@requires_db
async def test_chat_can_retrieve_from_an_uploaded_document(client) -> None:
    """The actual point of the feature: ask about something only the uploaded
    document could know, and confirm it shows up as a cited source — not just that
    the file was stored, but that it's genuinely searchable alongside the curated KB."""
    headers = await _auth_headers(client, "doc-rag@example.com")

    distinctive_fact = (
        "Project Zephyr is our internal codename for the Q3 checkout redesign. "
        "The Zephyr rollout increased mobile checkout completion by 12 percent."
    )
    upload = await client.post(
        "/api/documents/upload",
        headers=headers,
        files={"file": ("project-zephyr.txt", distinctive_fact.encode(), "text/plain")},
    )
    assert upload.status_code == 201

    chat = await client.post(
        "/api/chat/message", json={"message": "What was Project Zephyr and what impact did it have?"}, headers=headers
    )
    assert chat.status_code == 200
    body = chat.json()
    source_slugs = [s["slug"] for s in body["sources"]]
    assert any(slug.startswith("user-doc:") for slug in source_slugs), f"Expected a user-doc citation, got {source_slugs}"
