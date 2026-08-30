from tests.conftest import requires_db


async def _signup(client, email: str) -> dict[str, str]:
    resp = await client.post("/api/auth/signup", json={"email": email, "password": "hunter2pass", "persona": "learner"})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@requires_db
async def test_get_kb_document_by_real_slug_returns_full_content(client) -> None:
    headers = await _signup(client, "kb-doc-reader@example.com")

    # Any chat citation returns real slugs — grab one instead of hardcoding a title
    # that might change if the knowledge base content is edited.
    chat = await client.post("/api/chat/message", json={"message": "What does a p-value mean?"}, headers=headers)
    slug = chat.json()["sources"][0]["slug"]

    resp = await client.get(f"/api/kb/{slug}", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["slug"] == slug
    assert len(body["title"]) > 0
    assert len(body["content"]) > 0


@requires_db
async def test_get_kb_document_unknown_slug_returns_404(client) -> None:
    headers = await _signup(client, "kb-doc-unknown@example.com")
    resp = await client.get("/api/kb/does-not-exist", headers=headers)
    assert resp.status_code == 404


@requires_db
async def test_get_kb_document_requires_auth(client) -> None:
    resp = await client.get("/api/kb/some-slug")
    assert resp.status_code == 401
