"""PawCare+ HTTP API package.

Exposes the LangGraph pet-health assessment workflow as a REST API so it can be
consumed by the React frontend (or any other client). The heavy lifting still
lives in ``graph.assess_pet_health`` — this package is a thin, well-typed bridge.
"""
