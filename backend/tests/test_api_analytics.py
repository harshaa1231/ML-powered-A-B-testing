from tests.conftest import requires_db


async def _signup(client, email: str, persona: str = "business") -> dict[str, str]:
    resp = await client.post("/api/auth/signup", json={"email": email, "password": "hunter2pass", "persona": persona})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@requires_db
async def test_analytics_overview_empty_state_has_no_experiments(client) -> None:
    headers = await _signup(client, "analytics-empty@example.com")
    resp = await client.get("/api/analytics/overview", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_experiments"] == 0
    assert body["significance_rate"] == 0.0
    assert body["trend"] == []
    assert "No experiments yet" in body["ai_summary"]


@requires_db
async def test_analytics_overview_aggregates_real_experiments(client) -> None:
    headers = await _signup(client, "analytics-agg@example.com")

    # One significant, one not.
    await client.post(
        "/api/experiments/simple",
        json={
            "name": "Sig test",
            "metric_type": "conversion",
            "control_conversions": 100,
            "control_total": 1000,
            "treatment_conversions": 200,
            "treatment_total": 1000,
        },
        headers=headers,
    )
    await client.post(
        "/api/experiments/simple",
        json={
            "name": "Not sig test",
            "metric_type": "conversion",
            "control_conversions": 100,
            "control_total": 1000,
            "treatment_conversions": 102,
            "treatment_total": 1000,
        },
        headers=headers,
    )

    resp = await client.get("/api/analytics/overview", headers=headers)
    body = resp.json()
    assert body["total_experiments"] == 2
    assert body["significant_count"] == 1
    assert body["significance_rate"] == 0.5
    assert sum(t["count"] for t in body["trend"]) == 2
    assert "two_proportion_z" in body["test_type_breakdown"]
    # RAG actually ran: the trends narrative cites real knowledge-base content.
    assert len(body["ai_summary"]) > 0
    assert len(body["sources"]) > 0


@requires_db
async def test_analytics_overview_scoped_per_user(client) -> None:
    headers_a = await _signup(client, "analytics-a@example.com")
    headers_b = await _signup(client, "analytics-b@example.com")

    await client.post(
        "/api/experiments/simple",
        json={
            "name": "A's test",
            "metric_type": "conversion",
            "control_conversions": 50,
            "control_total": 500,
            "treatment_conversions": 60,
            "treatment_total": 500,
        },
        headers=headers_a,
    )

    resp_b = await client.get("/api/analytics/overview", headers=headers_b)
    assert resp_b.json()["total_experiments"] == 0


@requires_db
async def test_analytics_overview_works_for_learner_persona_too(client) -> None:
    """RAG-grounded analytics isn't business-only — a learner practicing on their own
    experiments gets the same grounded trends summary."""
    headers = await _signup(client, "analytics-learner@example.com", persona="learner")

    await client.post(
        "/api/experiments/simple",
        json={
            "name": "Practice run",
            "metric_type": "conversion",
            "control_conversions": 40,
            "control_total": 400,
            "treatment_conversions": 60,
            "treatment_total": 400,
        },
        headers=headers,
    )

    resp = await client.get("/api/analytics/overview", headers=headers)
    body = resp.json()
    assert body["total_experiments"] == 1
    assert len(body["ai_summary"]) > 0
    assert len(body["sources"]) > 0


@requires_db
async def test_analytics_overview_requires_auth(client) -> None:
    resp = await client.get("/api/analytics/overview")
    assert resp.status_code == 401
