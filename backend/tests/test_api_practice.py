from tests.conftest import requires_db


async def _signup(client, email: str, persona: str = "learner") -> dict[str, str]:
    resp = await client.post("/api/auth/signup", json={"email": email, "password": "hunter2pass", "persona": persona})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


SAMPLE_RESULTS = {
    "test_name": "Two-Proportion Z-Test",
    "p_value": 0.002,
    "effect_size": 3.1,
    "uplift_percentage": 18.0,
    "is_significant": True,
    "p_control": 0.1,
    "p_treatment": 0.118,
    "n_control": 1000,
    "n_treatment": 1000,
}


@requires_db
async def test_practice_feedback_is_rag_grounded_for_learner(client) -> None:
    headers = await _signup(client, "practice-learner@example.com", persona="learner")

    resp = await client.post(
        "/api/practice/feedback",
        json={
            "scenario_name": "Cookie Cats gate placement",
            "learner_conclusion": "I think the result is significant and treatment is clearly better.",
            "results": SAMPLE_RESULTS,
        },
        headers=headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["feedback"]) > 0
    # RAG actually ran: real chunks were retrieved and cited, not just a bare model call.
    assert len(body["sources"]) > 0
    assert all("slug" in s and "title" in s for s in body["sources"])


@requires_db
async def test_practice_feedback_also_works_for_business_persona(client) -> None:
    """Practice Lab isn't gated to the learner persona at the API level — a business
    account can use it too, and still gets grounded feedback."""
    headers = await _signup(client, "practice-business@example.com", persona="business")

    resp = await client.post(
        "/api/practice/feedback",
        json={
            "scenario_name": "Checkout button color",
            "learner_conclusion": "Not enough data to tell yet.",
            "results": SAMPLE_RESULTS,
        },
        headers=headers,
    )
    assert resp.status_code == 200
    assert len(resp.json()["feedback"]) > 0


@requires_db
async def test_practice_feedback_requires_auth(client) -> None:
    resp = await client.post(
        "/api/practice/feedback",
        json={"scenario_name": "x", "learner_conclusion": "y", "results": SAMPLE_RESULTS},
    )
    assert resp.status_code == 401
