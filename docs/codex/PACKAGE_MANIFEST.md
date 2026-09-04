# Package Manifest

Generated repo-ready Markdown execution pack for `todayoneul/HCX05_MIRAE_ASSET`.

## Verification basis

- Actual implementation baseline reviewed: `codex/task4-eval-context`
- Historical Task 4 tip reviewed: `9d6e256c622d8769b5ad34d305deb38284278e0e`
- Current pack intentionally renumbers post-implementation work as Task 5~13.
- Hashes below verify every other Markdown file in this package; re-hash after copying/editing into the repository.

| File | Bytes | SHA-256 |
|---|---:|---|
| `AGENTS.md` | 7985 | `079ce19e6555e081e330a766e64f503151921b44515f7f102bb1177826f27828` |
| `docs/codex/FINAL_RELEASE_CHECKLIST.md` | 3521 | `61f960ff3ee0a3e2ac7777b94875c9f1a6ed411ecdf5b65328943a05240f9f35` |
| `docs/codex/HUMAN_REVIEW_GUIDE.md` | 3644 | `60660bef14d0bf6276a79493cac7fe0c724bd0b6fa142ba959a594f02a52747f` |
| `docs/codex/README.md` | 1821 | `93d265cb50b975fbb1dfc014b58645ed4be98ec1cf3d2312d041b47e51ed7725` |
| `docs/codex/ROADMAP_TASK5_TO_SUBMISSION.md` | 5493 | `456e7dd913cf7296c41cd463e2559fb91756e413ddd5112d72dd2e3055372bb5` |
| `docs/codex/TASK_05A_REVIEW_WORKFLOW.md` | 5431 | `155908236326fbdfc623231299ea541085a74b3ff6386fe88334e6238d6f6761` |
| `docs/codex/TASK_05C_RETRIEVAL_BASELINE.md` | 2852 | `abd89b99173640213bdf3126d6eab0a93e62381d6e918e5a6c019c57c3da829e` |
| `docs/codex/TASK_06A_HCX_RUNTIME.md` | 3743 | `a5344a294cb8b9ad1185ae31676ee979cfa4a173e7da0d8d1e439dfd17350412` |
| `docs/codex/TASK_06B_TOOL_REGISTRY.md` | 2594 | `ef1d0c7cc6f569e73b344a1ac56ff97ca37136f91854a2fb32506798bf8d2beb` |
| `docs/codex/TASK_07_AGENT_V1.md` | 3883 | `8dcb59f8b08c4c533714c432c3a6e14eafe588225a0e64125649e7f4df0cbab3` |
| `docs/codex/TASK_08_GROUNDED_ANSWER.md` | 3621 | `3e78b5e77cf2b963e9bb8f11cb48e09eb881a51bd214705cf3006aa28a99292f` |
| `docs/codex/TASK_09_E2E_EVALUATION.md` | 2612 | `82c71cca0e8c52c1ae1d48ffa2b1f26ee63a2ba4f4008417457c6141c042fa2a` |
| `docs/codex/TASK_10A_LEXICAL_IMPROVEMENT.md` | 2141 | `9add6e9300a3e092fc28b15108a39a319832cf1c9e1ba882c21ad0072f4a9839` |
| `docs/codex/TASK_10B_HYBRID_POC.md` | 2590 | `a61e1a5b12e732bb300177475e4252eb2235b6692593bafb9b8da3381df5d60f` |
| `docs/codex/TASK_11_RELIABILITY.md` | 3307 | `9a4f4cd5f28ba40fce71492a0eabee8845ae2d1206fd6460172e98fa5e2b9d7c` |
| `docs/codex/TASK_12_SERVING_DEPLOYMENT.md` | 3353 | `bd00a45692a8a7add44e1b0479b335d7e70d545748db3c9f8f5ff3f6b91fde44` |
| `docs/codex/TASK_13_RELEASE_FREEZE.md` | 2971 | `ac94e58c126495365553175e8e54c72e880c6b6ead67429e6f35cf453dfa9cf5` |

## Sanity checks

- Root instruction filename is `AGENTS.md` (not `AEGENT.md` or `AGENT.md`).
- No `.env` value, API key, Authorization header, or user-specific absolute path is embedded.
- Holdout use is gated until Task 13 and explicitly isolated in implementation/evaluation tasks.
- HCX live calls are opt-in and Task 6 explicitly handles API-contract drift instead of freezing an old smoke observation.
- Task 10B is conditional and must not run without its entry/promotion gates.
- Generated candidate authority and human review authority are intentionally separated in Task 5A.
