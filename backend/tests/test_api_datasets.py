"""Regression coverage for a real statistical-integrity bug found during manual
verification: /api/datasets/samples/{key} used to cap every response at 2000
rows via df.head(2000). Sample datasets are our own small, fixed-size, bounded
data (the largest is ~90k rows) — truncating them silently throws away most of
the sample and can flip a genuinely significant result (Cookie Cats' real
effect is only ~0.8 points and needs its full sample size to reliably clear
significance) into a false negative. These tests pin down that the full
dataset is now returned, while the user-configurable, up-to-200k-row generator
endpoint still truncates its preview and says so honestly via `truncated`.
"""

from tests.conftest import requires_db


async def _signup(client, email: str) -> dict[str, str]:
    resp = await client.post("/api/auth/signup", json={"email": email, "password": "hunter2pass", "persona": "business"})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@requires_db
async def test_cookie_cats_sample_is_served_in_full_not_truncated(client) -> None:
    headers = await _signup(client, "datasets-cookiecats@example.com")

    summary = await client.get("/api/datasets/samples", headers=headers)
    assert summary.status_code == 200
    cookie_cats = next(s for s in summary.json() if s["key"] == "cookie_cats")
    assert cookie_cats["row_count"] == 90_189

    detail = await client.get("/api/datasets/samples/cookie_cats", headers=headers)
    assert detail.status_code == 200
    body = detail.json()
    assert body["row_count"] == 90_189
    # The historical bug: `rows` silently had far fewer entries than `row_count` claimed.
    assert len(body["rows"]) == 90_189


@requires_db
async def test_every_sample_dataset_rows_match_its_reported_row_count(client) -> None:
    headers = await _signup(client, "datasets-allsamples@example.com")

    summary = await client.get("/api/datasets/samples", headers=headers)
    for s in summary.json():
        detail = await client.get(f"/api/datasets/samples/{s['key']}", headers=headers)
        assert detail.status_code == 200
        body = detail.json()
        assert len(body["rows"]) == body["row_count"], f"{s['key']} rows don't match its row_count"


@requires_db
async def test_unknown_sample_dataset_key_returns_404(client) -> None:
    headers = await _signup(client, "datasets-unknown@example.com")
    resp = await client.get("/api/datasets/samples/does-not-exist", headers=headers)
    assert resp.status_code == 404


@requires_db
async def test_generator_still_truncates_large_previews_and_says_so(client) -> None:
    """The generator endpoint takes a user-specified n_samples up to 200,000 — unlike
    the fixed sample datasets, an uncapped preview there really could be tens of MB,
    so it keeps a cap but must honestly report truncation via the `truncated` flag."""
    headers = await _signup(client, "datasets-generator@example.com")

    resp = await client.post(
        "/api/datasets/generator/generate", json={"domain": "tech", "n_samples": 5000}, headers=headers
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["row_count"] == 5000
    assert len(body["rows"]) < body["row_count"]
    assert body["truncated"] is True
