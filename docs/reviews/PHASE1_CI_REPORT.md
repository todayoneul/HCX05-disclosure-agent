# Phase 1 CI Quality Gates Report

- Branch: `chore/phase1-ci-quality-gates`
- Base commit: `b1203dfc11c034ad0d2f7dfba216d31034409e51`
- Base branch: `opendart-origin/main`
- Report date: 2026-09-18 (Asia/Seoul)

## 1. Baseline

Phase 0 was fast-forward merged into `opendart-origin/main` before this branch was created. The local branch named `main` tracks a different repository, so this branch was created explicitly from `opendart-origin/main` to prevent cross-repository changes.

| Item | Result |
|---|---|
| Python | `3.13.11` in the project virtual environment |
| uv | `0.9.26` |
| `uv lock --check` | PASS |
| Locked dev/UI installation | PASS |
| Collection | 2,078 tests in 0.68 s |
| Full suite | 2,073 passed, 5 skipped, 0 failed in 4.91 s |
| `git diff --check` | PASS |
| Existing failures | None |

The five existing skips are unchanged: one corpus contract requires restored submission assets, and four real-context gates require an explicit immutable `--pipeline-root`. Normal tests use fake/queue/no-network transports and do not require HCX or OpenDART credentials.

## 2. Implemented Changes

### Workflow

`.github/workflows/ci.yml` runs on every pull request and on pushes to `main`. It defines two jobs:

1. `quality` verifies `uv.lock`, installs the locked development and UI dependency sets, and runs the complete offline pytest suite with `--no-sync`.
2. `docker` builds the production image from the Dockerfile after the quality job succeeds. It validates image construction only and does not start the credential-dependent service.

The workflow pins Python `3.13.11` and uv `0.9.26`, matching the Dockerfile and project constraints. Official GitHub Actions are pinned to immutable commits corresponding to `actions/checkout` v4.2.2 and `actions/setup-python` v5.6.0.

### Trigger configuration

```yaml
on:
  pull_request:
  push:
    branches:
      - main
```

### Commands automated

```text
uv lock --check
uv sync --locked --extra dev --extra ui
uv run --no-sync pytest -q
docker build -t disclosure-agent:ci .
```

No production source, API contract, OpenDART behavior, financial calculation, agent route, or runtime setting changed.

## 3. Security

- Workflow permissions are restricted to `contents: read`.
- `pull_request_target` is not used.
- No write, package, deployment, or identity-token permission is granted.
- `HCX_API_KEY` and `OPEN_DART` are not configured as CI secrets or environment variables.
- Normal test execution is offline and live-capable probes require explicit opt-in gates that CI does not provide.
- The Docker job builds the image without starting the service and therefore needs no runtime credentials.
- Action versions and uv are pinned; dependency installation is enforced by `uv.lock`.

## 4. Validation

### Local commands and results

```text
uv lock --check
  PASS — resolved 60 packages

uv sync --locked --extra dev --extra ui
  PASS — 57 packages audited

uv run --no-sync pytest --collect-only -q
  PASS — 2078 tests collected in 0.68s

uv run --no-sync pytest -q
  PASS — 2073 passed, 5 skipped in 4.91s

git diff --check
  PASS

docker build -t disclosure-agent:ci .
  NOT RUN — Docker CLI is unavailable in the local environment
```

### GitHub Actions

- Pull request: https://github.com/todayoneul/HCX05-disclosure-agent/pull/1
- Validation run: https://github.com/todayoneul/HCX05-disclosure-agent/actions/runs/35300944303
- Status: **PASSED**
- `Offline quality gates`: **PASSED**
- `Build production image`: **PASSED**

The run validated commit `b8a61ddf596d0215703ac543602e5b965e223c46`. GitHub Actions installed Python and uv, verified the lock, installed locked dev/UI dependencies, ran the offline suite, and built the production image without HCX or OpenDART credentials. The local Docker result remains `NOT RUN`; the successful Docker result above is from GitHub Actions.

## 5. Deferred Work

- Ruff adoption and any resulting source cleanup
- mypy or pyright adoption with a scoped legacy policy
- automated secret and dependency security scanning
- credentialed Docker runtime integration testing
- OpenDART cache persistence
- arrival-to-response deadline semantics
- bounded request backpressure
- unknown financial-basis handling
- AgentRunner modularization

These concerns remain separate because this PR automates existing quality gates only.
