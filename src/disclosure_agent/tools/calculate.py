from __future__ import annotations

import re
from decimal import (
    Decimal,
    DecimalException,
    InvalidOperation,
    ROUND_DOWN,
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    ROUND_UP,
)
from functools import cmp_to_key

from .common import error, result


ROUNDINGS = {
    "ROUND_HALF_UP": ROUND_HALF_UP,
    "ROUND_HALF_EVEN": ROUND_HALF_EVEN,
    "ROUND_DOWN": ROUND_DOWN,
    "ROUND_UP": ROUND_UP,
}
BINARY_OPERATIONS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "ratio_percent",
    "percent_change",
}
OPERATIONS = BINARY_OPERATIONS | {"sum", "rank_desc", "rank_ratio_desc"}
BINARY_VALIDATION_ERROR = (
    "calculation requires two decimal strings, scale 0..12, "
    "and an allowed rounding mode"
)
VARIABLE_VALIDATION_ERROR = (
    "calculation requires decimal strings, scale 0..12, "
    "and an allowed rounding mode"
)
DECIMAL_PATTERN = re.compile(
    r"^-?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?$"
)


def calculate(
    operation: str,
    inputs: list[str] | tuple[str, ...],
    *,
    scale: int = 2,
    rounding: str = "ROUND_HALF_UP",
) -> dict:
    if not isinstance(operation, str) or operation not in OPERATIONS:
        return error("unsupported calculation operation")
    if (
        not isinstance(inputs, (list, tuple))
        or not all(isinstance(value, str) for value in inputs)
        or isinstance(scale, bool)
        or not isinstance(scale, int)
        or not 0 <= scale <= 12
        or not isinstance(rounding, str)
        or rounding not in ROUNDINGS
    ):
        return error(
            BINARY_VALIDATION_ERROR
            if operation in BINARY_OPERATIONS
            else VARIABLE_VALIDATION_ERROR
        )
    if operation in BINARY_OPERATIONS and len(inputs) != 2:
        return error(BINARY_VALIDATION_ERROR)
    if operation == "sum" and not 1 <= len(inputs) <= 20:
        return error("sum requires 1..20 decimal strings")
    if operation == "rank_desc" and not 2 <= len(inputs) <= 10:
        return error("rank_desc requires 2..10 decimal strings")
    if operation == "rank_ratio_desc" and (
        not 4 <= len(inputs) <= 20 or len(inputs) % 2 != 0
    ):
        return error("rank_ratio_desc requires 2..10 numerator/denominator pairs")
    if not all(
        len(value) <= 100 and DECIMAL_PATTERN.fullmatch(value) is not None
        for value in inputs
    ):
        return error("inputs must be decimal strings")
    try:
        numbers = [Decimal(value.replace(",", "")) for value in inputs]
    except (InvalidOperation, TypeError, ValueError):
        return error("inputs must be decimal strings")
    if not all(number.is_finite() for number in numbers):
        return error("inputs must be finite")
    if operation == "rank_ratio_desc" and any(
        denominator <= 0 for denominator in numbers[1::2]
    ):
        return error("rank_ratio_desc denominators must be positive")
    ordered_indices = None
    try:
        if operation in BINARY_OPERATIONS:
            left, right = numbers
        if operation == "add":
            raw = left + right
        elif operation == "subtract":
            raw = left - right
        elif operation == "multiply":
            raw = left * right
        elif operation == "divide":
            raw = left / right
        elif operation == "ratio_percent":
            raw = left / right * 100
        elif operation == "percent_change":
            raw = (right - left) / left * 100
        elif operation == "sum":
            raw = sum(numbers, Decimal(0))
        elif operation == "rank_desc":
            ordered_indices = sorted(
                range(len(numbers)),
                key=numbers.__getitem__,
                reverse=True,
            )
            raw = numbers[ordered_indices[0]]
        else:
            pairs = tuple(zip(numbers[0::2], numbers[1::2], strict=True))

            def compare(left_index: int, right_index: int) -> int:
                left_numerator, left_denominator = pairs[left_index]
                right_numerator, right_denominator = pairs[right_index]
                left_cross = left_numerator * right_denominator
                right_cross = right_numerator * left_denominator
                if left_cross > right_cross:
                    return -1
                if left_cross < right_cross:
                    return 1
                return left_index - right_index

            ordered_indices = sorted(
                range(len(pairs)), key=cmp_to_key(compare)
            )
            numerator, denominator = pairs[ordered_indices[0]]
            raw = numerator / denominator * 100
    except (ZeroDivisionError, DecimalException):
        return error("invalid or out-of-range Decimal operation")
    try:
        quantized = raw.quantize(
            Decimal(1).scaleb(-scale), rounding=ROUNDINGS[rounding]
        )
    except (InvalidOperation, ValueError, OverflowError):
        return error("result exceeds supported Decimal range")
    data = {
        "operation": operation,
        "inputs": list(inputs),
        "scale": scale,
        "rounding": rounding,
        "result": format(quantized, "f"),
    }
    if ordered_indices is not None:
        data["ordered_indices"] = ordered_indices
    return result("ok", data)
