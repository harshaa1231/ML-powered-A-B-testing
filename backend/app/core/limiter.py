"""Shared rate limiter, split into its own module so both `app.main` (which
attaches it to the app and adds the middleware) and route modules (which
apply `@limiter.limit(...)` to individual endpoints) can import it without
a circular import through `app.main`.

In-memory storage is sufficient here: the free-tier deploy target runs a
single uvicorn process, not multiple workers/instances, so there's no need
for a shared external store (e.g. Redis) to keep counts consistent.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
