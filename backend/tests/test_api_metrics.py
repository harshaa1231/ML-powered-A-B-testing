from tests.conftest import requires_db


async def _auth_headers(client, email: str) -> dict[str, str]:
    resp = await client.post("/api/auth/signup", json={"email": email, "password": "hunter2pass", "persona": "business"})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@requires_db
async def test_create_and_list_metrics(client) -> None:
    headers = await _auth_headers(client, "metrics-creator@example.com")

    created = await client.post(
        "/api/metrics",
        json={"name": "Checkout Conversion", "description": "% of visitors who complete checkout", "column_name": "converted"},
        headers=headers,
    )
    assert created.status_code == 201
    assert created.json()["is_guardrail"] is False

    listed = await client.get("/api/metrics", headers=headers)
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert listed.json()[0]["name"] == "Checkout Conversion"


@requires_db
async def test_duplicate_metric_name_rejected(client) -> None:
    headers = await _auth_headers(client, "metrics-dupe@example.com")
    payload = {"name": "Page Load Time", "column_name": "load_time_ms", "is_guardrail": True}
    await client.post("/api/metrics", json=payload, headers=headers)
    dup = await client.post("/api/metrics", json=payload, headers=headers)
    assert dup.status_code == 409


@requires_db
async def test_same_metric_name_allowed_across_different_users(client) -> None:
    headers_a = await _auth_headers(client, "metrics-user-a@example.com")
    headers_b = await _auth_headers(client, "metrics-user-b@example.com")
    payload = {"name": "Conversion Rate", "column_name": "converted"}

    resp_a = await client.post("/api/metrics", json=payload, headers=headers_a)
    resp_b = await client.post("/api/metrics", json=payload, headers=headers_b)
    assert resp_a.status_code == 201
    assert resp_b.status_code == 201


@requires_db
async def test_delete_metric(client) -> None:
    headers = await _auth_headers(client, "metrics-deleter@example.com")
    created = await client.post(
        "/api/metrics", json={"name": "Error Rate", "column_name": "error_rate", "is_guardrail": True}, headers=headers
    )
    metric_id = created.json()["id"]

    deleted = await client.delete(f"/api/metrics/{metric_id}", headers=headers)
    assert deleted.status_code == 204

    listed = await client.get("/api/metrics", headers=headers)
    assert listed.json() == []


@requires_db
async def test_cannot_delete_another_users_metric(client) -> None:
    headers_owner = await _auth_headers(client, "metrics-owner@example.com")
    headers_other = await _auth_headers(client, "metrics-intruder@example.com")
    created = await client.post(
        "/api/metrics", json={"name": "Revenue Per User", "column_name": "revenue"}, headers=headers_owner
    )
    metric_id = created.json()["id"]

    resp = await client.delete(f"/api/metrics/{metric_id}", headers=headers_other)
    assert resp.status_code == 404


@requires_db
async def test_metrics_require_auth(client) -> None:
    resp = await client.get("/api/metrics")
    assert resp.status_code == 401
