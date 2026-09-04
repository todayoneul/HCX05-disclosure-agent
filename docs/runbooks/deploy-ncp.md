# Task 12 NCP deployment runbook

## Approval boundary

Do not create an NCP VM, allocate credit, change an ACG/firewall, expose a
public endpoint, or call HCX until the user explicitly approves that action.
Before approval, record the intended VM type, expected charge, region, public
IP plan, evaluator allowlist, and planned HCX smoke count.

Required secret: `HCX_API_KEY`. Optional non-secret model selector:
`HCX_MODEL` (default `HCX-005`). Never print either environment value, copy
`.env` into the image, or include an Authorization header in logs.

## Immutable inputs

- Git commit: record the approved Task 12 commit.
- Pipeline release: `50d9a055f0811118b902d9937f2359c866acc07d886183e90bd2697547ef88bd`
- Retrieval release: `3f16e81a440c47b899b6c48a4bfb1a88277967da92f3f7cabd0806c63a9506d5`
- Image: build from `Dockerfile`, then record the full `sha256:` image ID.
- Mount repository `artifacts/` and `data/` at `/app/artifacts:ro` and
  `/app/data:ro`. Do not use the legacy `pipeline/out` LFS objects.

## Local image gate

```sh
docker compose build --pull
docker compose up -d
docker compose ps
curl --fail --silent http://127.0.0.1:8080/healthz
docker image inspect disclosure-agent:task12-local --format '{{.Id}}'
docker compose down
```

The health lineage must match both immutable IDs above. Startup failure is a
release blocker; do not run in a degraded mode. `/healthz` causes zero HCX
calls. An approved local `/answer` smoke uses one HCX call.

## Approved VM deployment

1. In the NCP console create/confirm VPC, public subnet, server, and ACG in that
   order. The supplied workshop guide uses a High-CPU 2 vCPU/4 GB Linux server,
   20 GB initial storage, a new public IP, and a newly generated login key as an
   example. Treat that as a starting point: confirm measured image/artifact
   disk use and container RSS before purchasing the final size, and protect the
   login key outside the repository.
2. Transfer only the approved checkout/artifact releases. Create `.env` on the
   VM with owner-only permissions; never transfer it through Git or image
   layers.
3. Build the exact image and record Git commit, image ID, artifact release IDs,
   build time, and operator in the release record.
4. Start with `docker compose up -d`. Verify `/healthz` from the VM before any
   public rule is changed.
5. Change the compose port binding from `127.0.0.1:8080:8080` to the approved
   standard HTTP/HTTPS ingress only in the deployment-specific override. Keep
   one application worker.
6. After the organizer publishes evaluator source IPs, allow only those ranges
   on TCP 80/443 plus the minimum operator source IP on TCP 22. The workshop
   slide's `0.0.0.0/0` examples explain connectivity but are not the final
   least-privilege rule. Do not guess the evaluator allowlist.
7. From outside the VM, verify `/healthz`, then—with explicit HCX approval—send
   one URL-encoded Korean `/answer` request. Expected HCX calls: one normally,
   plus only the bounded internal retries documented by Task 11 if a temporary
   transport failure occurs.
8. Replace the README submission placeholder only after the public request is
   verified. Record the exact endpoint without query values.

The official endpoint has no authentication header and returns exactly five
string fields. The evaluator is sequential, waits 300 seconds, and may retry a
timeout or 5xx response twice; the server hard deadline remains 270 seconds.

## Rollback

Keep the prior approved image ID and immutable artifact releases. Roll back by
stopping the failed container and starting the prior image with the same
read-only mounts. Do not mutate a release directory or its current pointer
during rollback. Follow `restart.md` for evidence capture.
