import pytest

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


def test_sum_accepts_one_to_twenty_inputs_with_existing_result_conventions():
    single = calculate("sum", ("1.245",), scale=2, rounding="ROUND_HALF_EVEN")
    assert single["status"] == "ok"
    assert single["data"] == {
        "operation": "sum",
        "inputs": ["1.245"],
        "scale": 2,
        "rounding": "ROUND_HALF_EVEN",
        "result": "1.24",
    }

    twenty = calculate("sum", ["0.1"] * 20, scale=1)
    assert twenty["status"] == "ok"
    assert twenty["data"]["result"] == "2.0"


def test_calculation_accepts_schema_valid_grouped_decimal_strings():
    summed = calculate("sum", ["1,000", "2,500"], scale=0)

    assert summed["status"] == "ok"
    assert summed["data"]["inputs"] == ["1,000", "2,500"]
    assert summed["data"]["result"] == "3500"


def test_rank_desc_compares_exact_decimals_and_uses_stable_original_indices():
    ranked = calculate("rank_desc", ["1.004", "1.003", "1.0040", "2.000"], scale=2)
    assert ranked["status"] == "ok"
    assert ranked["data"] == {
        "operation": "rank_desc",
        "inputs": ["1.004", "1.003", "1.0040", "2.000"],
        "scale": 2,
        "rounding": "ROUND_HALF_UP",
        "result": "2.00",
        "ordered_indices": [3, 0, 2, 1],
    }


def test_rank_desc_formats_the_exact_top_value_with_requested_rounding():
    ranked = calculate("rank_desc", ["1.005", "1.004"], scale=2, rounding="ROUND_HALF_UP")
    assert ranked["data"]["result"] == "1.01"
    assert ranked["data"]["ordered_indices"] == [0, 1]

    ten_inputs = [str(value) for value in range(10)]
    assert calculate("rank_desc", ten_inputs)["data"]["ordered_indices"] == list(range(9, -1, -1))


def test_rank_ratio_desc_compares_unrounded_ratios_and_preserves_stable_ties():
    ranked = calculate(
        "rank_ratio_desc",
        ["100049", "1000000", "100041", "1000000", "1", "2", "2", "4"],
        scale=2,
    )

    assert ranked["status"] == "ok"
    assert ranked["data"] == {
        "operation": "rank_ratio_desc",
        "inputs": [
            "100049",
            "1000000",
            "100041",
            "1000000",
            "1",
            "2",
            "2",
            "4",
        ],
        "scale": 2,
        "rounding": "ROUND_HALF_UP",
        "result": "50.00",
        "ordered_indices": [2, 3, 0, 1],
    }


@pytest.mark.parametrize(
    "inputs",
    [
        ["1", "2", "3"],
        ["1", "2"] * 11,
        ["1", "0", "2", "3"],
        ["1", "-2", "2", "3"],
    ],
)
def test_rank_ratio_desc_rejects_invalid_pair_contract(inputs):
    assert calculate("rank_ratio_desc", inputs)["status"] == "error"


@pytest.mark.parametrize("operation", ["add", "subtract", "multiply", "divide", "ratio_percent", "percent_change"])
@pytest.mark.parametrize("inputs", [[], ["1"], ["1", "2", "3"]])
def test_existing_binary_operations_still_require_exactly_two_inputs(operation, inputs):
    assert calculate(operation, inputs)["status"] == "error"


@pytest.mark.parametrize(
    ("operation", "inputs"),
    [
        ("sum", []),
        ("sum", ["1"] * 21),
        ("rank_desc", ["1"]),
        ("rank_desc", ["1"] * 11),
        ("rank_ratio_desc", ["1", "2"]),
        ("rank_ratio_desc", ["1", "2", "3"]),
        ("rank_ratio_desc", ["1", "2"] * 11),
    ],
)
def test_variable_arity_operations_reject_wrong_input_lengths(operation, inputs):
    assert calculate(operation, inputs)["status"] == "error"


@pytest.mark.parametrize(
    "operation,valid_inputs",
    [
        ("sum", ["1"]),
        ("rank_desc", ["1", "2"]),
        ("rank_ratio_desc", ["1", "2", "3", "4"]),
    ],
)
@pytest.mark.parametrize("invalid_value", [True, 1, "not-a-decimal", "NaN", "Infinity", "-Infinity"])
def test_new_operations_reject_non_strings_malformed_and_nonfinite_values(operation, valid_inputs, invalid_value):
    inputs = [*valid_inputs, invalid_value]
    if operation == "rank_desc":
        inputs = inputs[-2:]
    assert calculate(operation, inputs)["status"] == "error"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale": -1},
        {"scale": 13},
        {"scale": True},
        {"scale": 2.0},
        {"rounding": "ROUND_CEILING"},
        {"rounding": True},
    ],
)
@pytest.mark.parametrize(
    "operation,inputs",
    [
        ("sum", ["1"]),
        ("rank_desc", ["1", "2"]),
        ("rank_ratio_desc", ["1", "2", "3", "4"]),
    ],
)
def test_new_operations_reject_invalid_scale_and_rounding(operation, inputs, kwargs):
    assert calculate(operation, inputs, **kwargs)["status"] == "error"


@pytest.mark.parametrize("operation", ["average", "rank_asc", "eval", True])
def test_calculation_rejects_every_unsupported_operation(operation):
    assert calculate(operation, ["1", "2"])["status"] == "error"
