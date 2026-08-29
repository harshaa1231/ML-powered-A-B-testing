from tests.conftest import requires_db


@requires_db
async def test_signup_then_me(client) -> None:
    signup = await client.post(
        "/api/auth/signup", json={"email": "alice@example.com", "password": "hunter2pass", "persona": "business"}
    )
    assert signup.status_code == 201
    token = signup.json()["access_token"]

    me = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "alice@example.com"
    assert me.json()["persona"] == "business"


@requires_db
async def test_signup_duplicate_email_and_persona_rejected(client) -> None:
    await client.post("/api/auth/signup", json={"email": "bob@example.com", "password": "hunter2pass", "persona": "business"})
    dup = await client.post("/api/auth/signup", json={"email": "bob@example.com", "password": "different", "persona": "business"})
    assert dup.status_code == 409


@requires_db
async def test_same_email_can_hold_a_business_and_a_learner_account(client) -> None:
    """The account identity is (email, persona), not email alone — the same person can
    have a business account and a learner account under one email, as two fully
    separate accounts (own password, own data)."""
    business = await client.post(
        "/api/auth/signup", json={"email": "dana@example.com", "password": "business-pass", "persona": "business"}
    )
    learner = await client.post(
        "/api/auth/signup", json={"email": "dana@example.com", "password": "learner-pass", "persona": "learner"}
    )
    assert business.status_code == 201
    assert learner.status_code == 201
    assert business.json()["access_token"] != learner.json()["access_token"]

    business_me = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {business.json()['access_token']}"})
    learner_me = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {learner.json()['access_token']}"})
    assert business_me.json()["persona"] == "business"
    assert learner_me.json()["persona"] == "learner"
    assert business_me.json()["id"] != learner_me.json()["id"]


@requires_db
async def test_login_with_wrong_password_rejected(client) -> None:
    await client.post(
        "/api/auth/signup", json={"email": "carol@example.com", "password": "correct-password", "persona": "business"}
    )
    login = await client.post(
        "/api/auth/login", json={"email": "carol@example.com", "password": "wrong-password", "persona": "business"}
    )
    assert login.status_code == 401


@requires_db
async def test_login_with_wrong_persona_for_existing_account_rejected(client) -> None:
    """Correct email + correct password, but for the OTHER persona's account — since
    that specific (email, persona) account doesn't exist, this must fail rather than
    silently logging into the wrong account."""
    await client.post(
        "/api/auth/signup", json={"email": "frankie@example.com", "password": "correct-password", "persona": "business"}
    )
    login = await client.post(
        "/api/auth/login", json={"email": "frankie@example.com", "password": "correct-password", "persona": "learner"}
    )
    assert login.status_code == 401


@requires_db
async def test_me_requires_auth(client) -> None:
    resp = await client.get("/api/auth/me")
    assert resp.status_code == 401
