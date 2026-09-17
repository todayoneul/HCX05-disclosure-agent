"""Local chat UI layer for the OpenDART disclosure agent.

This package is separate from the FastAPI service so the same UI can point at
different backend work-trees by base URL, and can run in an offline mock mode
for testing without live HCX or OpenDART calls.
"""

__all__ = ["api_client", "conversation"]
