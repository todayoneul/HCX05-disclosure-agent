# Official Evaluation API Contract

Status: organizer guidance received on 2026-08-29 and endpoint submission
format reconfirmed on 2026-09-01. This document supersedes
older repository notes that treated HTTP/HTTPS or the serving window as wholly
unannounced. The exact operating dates and evaluator source IP range remain
pending organizer notice.

## Submission endpoint

- The 2026-09-01 announcement requires the submitted URL itself to have the
  form `http://12.34.56.78/answer` (illustrative IP only): no query string,
  surrounding quote, trailing explanation, or health path is submitted.
- The Agent must be reachable from the public internet on Naver Cloud or other server resources.
- The final endpoint URL must be written in the repository root `README.md`.
- The fixed request path is `/answer`.
- A public IP is sufficient; a domain or subdomain is not required.
- HTTP is the default on port 80.
- HTTPS is optional on port 443; a self-signed certificate is allowed.
- A personal machine may use a stable public tunnel URL such as ngrok.
- No authentication header or other request header is sent by the evaluator.
- When access control is desired, use the evaluator source-IP allowlist after the organizer publishes that range.

The README currently contains an explicit not-deployed placeholder. It must not
be mistaken for a valid submission URL and must be replaced only after Task 12
public-network verification.

## Request

```http
GET {team-end-point}/answer?question_id={question-id}&question={url-encoded-question}
```

Required query parameters:

- `question_id`: organizer-provided question identifier.
- `question`: the original private evaluation question.

The service must preserve both strings in the response. Unicode and URL
encoding are part of the serving contract.

## Response

Return `Content-Type: application/json` and one JSON object with these five
fields:

```json
{
  "question_id": "Q-001",
  "question": "평가 질의 원문",
  "retrieved_context": "답변 생성에 참고한 검색 문서",
  "think_trace": "고수준 사고·추론·도구 사용 감사 요약",
  "answer": "최종 생성 답변"
}
```

Contract rules:

- Every field value must be a JSON string, including empty results.
- `retrieved_context` may concatenate multiple documents using an arbitrary string delimiter.
- Field lengths have no formal limit, but excessively long output may be truncated or ignored by the evaluation system.
- `think_trace` must remain the repository-defined high-level audit summary. It must not expose hidden chain-of-thought, prompts, credentials, or authorization data.
- The deterministic response validator must reject missing, extra, or non-string fields before serving.

## Evaluator behavior

- Requests are sequential: one question at a time per team, with no concurrent evaluator requests.
- The evaluator waits up to 300 seconds for each request.
- A timeout or 5xx response is retried at most two times.
- The service must therefore be idempotent for the same `question_id` and `question` and must not multiply unbounded internal retries.
- The project retains a 270-second internal hard deadline to leave serialization and network margin before the external timeout.

## Operations

- The server operation window falls within `09.07~09.20` and will be announced separately.
- Each topic's operation period will be at most one week.
- The exact operation dates and evaluator source IP range are still pending.
- Restarting after submission for an unavoidable server fault is not a disqualification when it is not used to change the submitted result.
- Runtime logs must redact secrets and avoid retaining unnecessary full private questions or source text.

## Implementation gates

- Task 6A: HCX transport and opt-in model contract probe; no serving endpoint.
- Task 11: 270-second internal deadline, bounded retry/cache, and idempotence.
- Task 12: local API contract, public deployment, port/firewall, and README endpoint replacement.
- Task 13: final endpoint/checksum freeze and exact operating-notice verification.

No example IP or placeholder is a valid submission endpoint. NCP creation,
public firewall changes, tunnel activation, HCX calls, and credit consumption
remain explicit user-approved operations.

## Code submission repository

- Assigned repository: `https://github.com/miraeasset-aifestival-2026-dart/dis-099`
- Confirmed 2026-09-01: current local Git credential has read access and a
  no-write `git push --dry-run` reports write permission.
- A separate PAT application is not currently required. Recheck access before
  the final push and never record the active credential or token value.
