"""Regression coverage for a real gap found during manual testing: chat history was
persisted correctly all along, but nothing ever fetched it back — every page load
(and every login) started from a blank chat with no way to resume a prior
conversation. `/api/chat/sessions/latest` is what the frontend now calls on mount
to restore it."""

from tests.conftest import requires_db


async def _signup(client, email: str) -> dict[str, str]:
    resp = await client.post("/api/auth/signup", json={"email": email, "password": "hunter2pass", "persona": "learner"})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@requires_db
async def test_latest_session_is_empty_for_a_brand_new_account(client) -> None:
    headers = await _signup(client, "chat-fresh@example.com")
    resp = await client.get("/api/chat/sessions/latest", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] is None
    assert body["messages"] == []


@requires_db
async def test_latest_session_resumes_after_relogin(client) -> None:
    """Simulates the real bug report: send a message, then act like a brand-new
    login (fresh headers derived from a fresh login call, not the signup token) and
    confirm the conversation is still there."""
    email = "chat-resume@example.com"
    signup = await client.post("/api/auth/signup", json={"email": email, "password": "hunter2pass", "persona": "learner"})
    headers = {"Authorization": f"Bearer {signup.json()['access_token']}"}

    sent = await client.post("/api/chat/message", json={"message": "What is a p-value?"}, headers=headers)
    assert sent.status_code == 200
    session_id = sent.json()["session_id"]

    login = await client.post("/api/auth/login", json={"email": email, "password": "hunter2pass", "persona": "learner"})
    relogin_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    resp = await client.get("/api/chat/sessions/latest", headers=relogin_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == session_id
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "What is a p-value?"
    assert body["messages"][1]["role"] == "assistant"
    assert len(body["messages"][1]["sources"]) > 0


@requires_db
async def test_latest_session_scoped_per_user(client) -> None:
    headers_a = await _signup(client, "chat-user-a@example.com")
    headers_b = await _signup(client, "chat-user-b@example.com")

    await client.post("/api/chat/message", json={"message": "Hello from user A"}, headers=headers_a)

    resp_b = await client.get("/api/chat/sessions/latest", headers=headers_b)
    assert resp_b.json()["session_id"] is None


@requires_db
async def test_latest_session_requires_auth(client) -> None:
    resp = await client.get("/api/chat/sessions/latest")
    assert resp.status_code == 401
