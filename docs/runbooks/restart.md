# Serving restart runbook

An unavoidable restart restores the same approved Git commit, image digest,
artifact release IDs, model contract, and prompt/config version. It must not be
used to change a submission result after freeze.

## Before restart

```sh
docker compose ps
docker image inspect disclosure-agent:task12-local --format '{{.Id}}'
docker compose logs --since 15m --no-color
```

Store redacted output only. Never copy `.env`, request text, API keys, or
Authorization headers into an incident record. Confirm the artifact mounts are
read-only and the expected current pointers have not changed.

## Graceful restart

```sh
docker compose restart --timeout 300 disclosure-agent
docker compose ps
curl --fail --silent http://127.0.0.1:8080/healthz
```

The five-minute grace window lets the one active request finish. If health does
not return both approved release IDs, stop and investigate; do not expose a
partially initialized service. A health check makes zero HCX calls.

## Recovery

If the current image cannot become healthy, start the prior approved digest
with the unchanged read-only artifacts, verify internal health, then restore
the approved ingress. Record reason, timestamps, image IDs, release IDs, and
whether any evaluator request could have been interrupted.
