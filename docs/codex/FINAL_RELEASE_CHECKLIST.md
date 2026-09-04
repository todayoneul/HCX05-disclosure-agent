# Final Release Checklist

## A. Source / Git

- [ ] 제출 저장소 `https://github.com/miraeasset-aifestival-2026-dart/dis-099` 접근 재확인
- [ ] 제출 저장소 `main`에 승인된 최소 패키지만 존재
- [ ] final commit SHA 기록
- [ ] `git status --short` clean
- [ ] `git diff --check` PASS
- [ ] 마감 후 변경 금지 절차 확인
- [ ] submission tag/commit과 제출 manifest 일치

## B. Environment

- [ ] Python version 기록
- [ ] `uv.lock` 존재 및 SHA-256 기록
- [ ] `uv lock --check` PASS
- [ ] clean environment에서 sync 가능
- [ ] `.env`/secret 미커밋
- [ ] Docker image에 secret 없음

## C. Corpus / Pipeline

- [ ] corpus immutable contract PASS
- [ ] pipeline current pointer verify
- [ ] pipeline release ID 기록
- [ ] SQLite integrity/foreign key check 필요 범위 PASS
- [ ] legacy artifact 수정 없음

## D. Retrieval

- [ ] retrieval pointer verify
- [ ] retrieval→pipeline lineage 정확
- [ ] retrieval release ID 기록
- [ ] baseline report/hash 기록
- [ ] latest/correction metadata regression 0

## E. Human Review / Evaluation

- [ ] generated candidate와 human review authority 분리 확인
- [ ] dev/regression human decision 검증
- [ ] stale review lineage 거부 test
- [ ] source anchor validation 0 failures
- [ ] holdout가 release-candidate 전 튜닝에 사용되지 않음
- [ ] holdout 실행 reason 기록

## F. Agent

- [ ] tool loop bounded
- [ ] model call bounded
- [ ] calculate tool 사용
- [ ] ContextPacker 사용
- [ ] no evidence → info-limit
- [ ] correction policy
- [ ] external/future/investment guard
- [ ] prompt injection fixture PASS
- [ ] secret/system prompt leakage 0

## G. Response Contract

- [ ] `question_id` string
- [ ] `question` string
- [ ] `retrieved_context` string
- [ ] `think_trace` string
- [ ] `answer` string
- [ ] answerable citation 100%
- [ ] correction-required mention 100%
- [ ] grounding 밖 핵심 claim 0

## H. Runtime / Faults

- [ ] 270s internal hard deadline
- [ ] 429 behavior
- [ ] 5xx behavior
- [ ] network timeout behavior
- [ ] bounded internal retry
- [ ] same request consistency
- [ ] artifact mismatch fail closed
- [ ] log secret redaction

## I. Local Server

- [ ] clean Docker build
- [ ] container startup
- [ ] `/answer` valid request
- [ ] Korean URL encoding
- [ ] info-limit request
- [ ] correction request
- [ ] repeated request
- [ ] health endpoint
- [ ] graceful restart

## J. NCP / Public Endpoint

- [ ] README endpoint가 query 없는 `http://<public-ip>/answer` 또는 승인된 HTTPS 동형식
- [ ] latest official HTTP/HTTPS rule 확인
- [ ] latest official serving period 확인
- [ ] public endpoint reachable
- [ ] response JSON decode
- [ ] sequential request test
- [ ] timeout 0
- [ ] image immutable digest 기록
- [ ] artifact mount read-only
- [ ] restart policy
- [ ] credit/budget 확인

## K. Evaluation

- [ ] dev/regression final run
- [ ] retrieval Recall@10 기록
- [ ] Closed fact/numeric accuracy 기록
- [ ] required fact coverage 기록
- [ ] citation compliance 기록
- [ ] correction compliance 기록
- [ ] info-limit/safety 기록
- [ ] failure taxonomy 기록
- [ ] holdout release-candidate run 기록
- [ ] latency/usage report deterministic payload와 분리

## L. Submission Package

- [ ] README
- [ ] API spec
- [ ] technical proposal
- [ ] release manifest
- [ ] checksums
- [ ] preprocessing artifacts/download link
- [ ] endpoint
- [ ] env var names
- [ ] runbook
- [ ] known limitations
- [ ] 외부에서 링크/endpoint 열림 확인

## M. Freeze

- [ ] 제출 목표 시각 전에 완료
- [ ] 공식 제출 화면 상태 확인
- [ ] freeze 시각 기록
- [ ] 이후 code/prompt/index/config/image 변경 금지
- [ ] 장애 복구는 동일 digest restart만
