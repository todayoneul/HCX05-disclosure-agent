# Serving monitoring runbook

Monitor availability and latency without storing raw questions or answers.
Application logs contain only a truncated request hash, status, duration, and
pipeline/retrieval release IDs. Treat even the hash as operational data and
retain it only for the evaluation window and incident review.

## Checks

- Probe `/healthz` locally every 30 seconds. It must be HTTP 200, `ready=true`,
  and expose the two approved release IDs.
- Alert on container unhealthy/restart, repeated 503 responses, requests near
  the 270-second deadline, disk pressure, memory pressure, and clock drift.
- Confirm a single Uvicorn worker and one in-process answer at a time.
- Keep Docker stdout/stderr rotation bounded. Do not enable request access logs
  that include query strings at the proxy or load balancer.

Useful redacted commands:

```sh
docker compose ps
docker compose logs --since 10m --no-color disclosure-agent
docker stats --no-stream disclosure-agent
curl --fail --silent http://127.0.0.1:8080/healthz
```

## Incident threshold

One isolated temporary 503 can be handled by the evaluator's bounded retry.
Repeated 503, lineage mismatch, unhealthy startup, or overlapping work is an
incident. Preserve redacted evidence, use `restart.md`, and do not change code,
prompts, artifacts, firewall scope, or the public endpoint during freeze
without the applicable human approval.
