from disclosure_agent.tools.calculate import calculate


def test_decimal_calculation_and_half_up_rounding():
    result = calculate("percent_change", ["100", "112.345"], scale=2)
    assert result["status"] == "ok"
    assert result["data"] == {"operation": "percent_change", "inputs": ["100", "112.345"], "scale": 2, "rounding": "ROUND_HALF_UP", "result": "12.35"}


def test_calculation_rejects_expressions_nonfinite_and_zero_division():
    assert calculate("add", ["1+2", "3"])["status"] == "error"
    assert calculate("add", ["NaN", "3"])["status"] == "error"
    assert calculate("divide", ["1", "0"])["status"] == "error"
    assert calculate("eval", ["1", "2"])["status"] == "error"


def test_non_division_operations_do_not_evaluate_division_branch():
    assert calculate("add", ["2", "0"])["data"]["result"] == "2.00"
    assert calculate("add", "12")["status"] == "error"
    assert calculate("add", [1.2, "3"])["status"] == "error"
    assert calculate("add", ["1", "2"], scale=True)["status"] == "error"
    assert calculate("multiply", ["1e999999", "1e999999"])["status"] == "error"
