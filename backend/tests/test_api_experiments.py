from tests.conftest import requires_db


async def _auth_headers(client, email: str) -> dict[str, str]:
    resp = await client.post("/api/auth/signup", json={"email": email, "password": "hunter2pass"})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@requires_db
async def test_decision_can_be_set_and_cleared(client) -> None:
    headers = await _auth_headers(client, "decision-maker@example.com")
    created = await client.post(
        "/api/experiments/simple",
        json={
            "name": "Checkout button color",
            "metric_type": "conversion",
            "control_conversions": 100,
            "control_total": 1000,
            "treatment_conversions": 140,
            "treatment_total": 1000,
        },
        headers=headers,
    )
    experiment_id = created.json()["id"]
    assert created.json()["decision"] is None

    shipped = await client.patch(f"/api/experiments/{experiment_id}/decision", json={"decision": "shipped"}, headers=headers)
    assert shipped.status_code == 200
    assert shipped.json()["decision"] == "shipped"

    cleared = await client.patch(f"/api/experiments/{experiment_id}/decision", json={"decision": None}, headers=headers)
    assert cleared.json()["decision"] is None


@requires_db
async def test_decision_requires_owning_the_experiment(client) -> None:
    owner_headers = await _auth_headers(client, "decision-owner@example.com")
    other_headers = await _auth_headers(client, "decision-other@example.com")
    created = await client.post(
        "/api/experiments/simple",
        json={
            "name": "Someone else's test",
            "metric_type": "conversion",
            "control_conversions": 100,
            "control_total": 1000,
            "treatment_conversions": 140,
            "treatment_total": 1000,
        },
        headers=owner_headers,
    )
    experiment_id = created.json()["id"]

    resp = await client.patch(
        f"/api/experiments/{experiment_id}/decision", json={"decision": "shipped"}, headers=other_headers
    )
    assert resp.status_code == 404


@requires_db
async def test_simple_conversion_test_creates_experiment(client) -> None:
    headers = await _auth_headers(client, "dave@example.com")

    resp = await client.post(
        "/api/experiments/simple",
        json={
            "name": "Signup button color",
            "metric_type": "conversion",
            "control_conversions": 100,
            "control_total": 1000,
            "treatment_conversions": 140,
            "treatment_total": 1000,
        },
        headers=headers,
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["mode"] == "simple"
    assert body["results"]["is_significant"] is True


@requires_db
async def test_advanced_test_auto_detects_ttest(client) -> None:
    headers = await _auth_headers(client, "erin@example.com")

    rows = [{"group": "control", "revenue": 40 + i % 5} for i in range(100)] + [
        {"group": "treatment", "revenue": 55 + i % 5} for i in range(100)
    ]
    resp = await client.post(
        "/api/experiments/advanced",
        json={"name": "Pricing test", "group_col": "group", "metric_col": "revenue", "test_type": "auto", "rows": rows},
        headers=headers,
    )
    assert resp.status_code == 201
    assert resp.json()["results"]["test_name"] == "Welch's t-test"


@requires_db
async def test_experiments_are_scoped_per_user(client) -> None:
    headers_a = await _auth_headers(client, "frank@example.com")
    headers_b = await _auth_headers(client, "grace@example.com")

    await client.post(
        "/api/experiments/simple",
        json={
            "name": "Frank's test",
            "metric_type": "conversion",
            "control_conversions": 10,
            "control_total": 100,
            "treatment_conversions": 15,
            "treatment_total": 100,
        },
        headers=headers_a,
    )

    listing = await client.get("/api/experiments", headers=headers_b)
    assert listing.status_code == 200
    assert listing.json() == []
