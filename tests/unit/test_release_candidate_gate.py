import pytest

from scripts import evaluate_release_candidate as gate


@pytest.mark.parametrize("args", [[], ["--live"], ["--reason", gate.LIVE_REASON],
    ["--live", "--reason", "wrong"]])
def test_requires_explicit_opt_in(args):
    with pytest.raises(SystemExit):
        gate.main(["--mode", "endpoint", *args])


@pytest.mark.parametrize("payload", [None, [], {}, {"answer": "fake"},
    dict.fromkeys(("question_id", "question", "retrieved_context", "think_trace", "answer"), 1)])
def test_rejects_malformed_payload(payload):
    assert not all(gate.check_response(gate.CASES[0], payload, "test").values())


def test_unknown_trace_does_not_become_success():
    case = next(case for case in gate.CASES if case.kind == "trap")
    payload = {"question_id": "test", "question": case.question, "retrieved_context": "",
               "think_trace": "unknown", "answer": "공시에서 확인할 수 없습니다."}
    checks = gate.check_response(case, payload, "test")
    assert not checks["expected_outcome"]


def test_request_binding():
    payload = {"question_id": "wrong", "question": "wrong", "retrieved_context": "",
               "think_trace": "unknown", "answer": ""}
    assert not gate.check_response(gate.CASES[0], payload, "test")["request_binding"]
