# HCX Native v3 Contract Decision — 2026-08-29

## Authority and scope

- 사용자가 비제출 테스트 계정 A로 정확히 4회의 Native v3 contract probe를 승인했다.
- 실행 코드는 branch `codex/task6-hcx-runtime`, commit
  `b5bf32a0d812b40791fecb3b7792d5e76a289141`의
  `scripts/probe_hcx_contract.py`였다.
- Function Calling 세 요청과 invalid payload 한 요청을 순차 실행했다.
- retry, embedding, OpenAI-compatible API, NCP resource, deployment 호출은 없었다.
- 계정 식별자, key/header, prompt, response body는 기록하지 않았다.

## Sanitized observations

| Probe mode | HTTP | Native status | toolCalls | arguments type | Outcome |
|---|---:|---:|---:|---|---|
| token limit omitted | 200 | 20000 | 1 | object | success |
| `maxTokens=1024` | 200 | 20000 | 1 | object | success |
| `maxTokens=2048` | 200 | 20000 | 1 | object | success |
| invalid payload | 400 | 40000 | not retained | not retained | expected rejection |

## Production decision

- Native v3 요청에 tools가 있고 caller가 token limit를 명시하지 않으면
  `maxTokens=1024`를 사용한다.
- tools가 없는 일반 chat 요청은 caller가 명시하지 않는 한 token-limit field를
  보내지 않는다.
- explicit `TokenLimit.omit()`과 `TokenLimit.max_tokens(...)`는 diagnostic 또는
  별도 승인 실험을 위해 유지하지만 production Function Calling 기본값은 아니다.

선택 이유는 세 mode가 모두 실제 Function Calling에 성공한 상태에서, 1024가 공식
Function Calling 최소 범위와 일치하고 2048보다 생성량, 비용, 지연 상한을 낮추는
명시적 값이기 때문이다. omit mode의 성공은 호환 관찰로만 보존한다.

## What remains unproven

- 다른 모델 또는 tuning model의 Function Calling 호환성
- OpenAI-compatible transport와 streaming contract
- embedding API 및 hybrid retrieval 승격 효과
- `Retry-After` HTTP-date 형식
- 실제 token 사용량과 청구액
- NCP 제출 계정 B와 public serving 환경

모델, 계정 또는 API 경로가 바뀌면 이 결정을 자동 승계하지 않고 별도 승인된 bounded
probe 또는 공식 계약 검증을 거친다.
