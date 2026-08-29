
from tests.conftest import requires_db


@requires_db
async def test_signup_then_me(client) -> None:
    signup = await client.post("/api/auth/signup", json={"email": "alice@example.com", "password": "hunter2pass"})
    assert signup.status_code == 201
    token = signup.json()["access_token"]

    me = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "alice@example.com"


@requires_db
async def test_signup_duplicate_email_rejected(client) -> None:
    await client.post("/api/auth/signup", json={"email": "bob@example.com", "password": "hunter2pass"})
    dup = await client.post("/api/auth/signup", json={"email": "bob@example.com", "password": "different"})
    assert dup.status_code == 409


@requires_db
async def test_login_with_wrong_password_rejected(client) -> None:
    await client.post("/api/auth/signup", json={"email": "carol@example.com", "password": "correct-password"})
    login = await client.post("/api/auth/login", json={"email": "carol@example.com", "password": "wrong-password"})
    assert login.status_code == 401


@requires_db
async def test_me_requires_auth(client) -> None:
    resp = await client.get("/api/auth/me")
    assert resp.status_code == 401
