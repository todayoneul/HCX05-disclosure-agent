from __future__ import annotations

from decimal import Decimal, DecimalException, InvalidOperation, ROUND_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, ROUND_UP

from .common import error, result


ROUNDINGS = {"ROUND_HALF_UP": ROUND_HALF_UP, "ROUND_HALF_EVEN": ROUND_HALF_EVEN, "ROUND_DOWN": ROUND_DOWN, "ROUND_UP": ROUND_UP}


def calculate(operation: str, inputs: list[str] | tuple[str, ...], *, scale: int = 2, rounding: str = "ROUND_HALF_UP") -> dict:
    if operation not in {"add", "subtract", "multiply", "divide", "ratio_percent", "percent_change"}:
        return error("unsupported calculation operation")
    if not isinstance(inputs, (list, tuple)) or not all(isinstance(value, str) for value in inputs) or isinstance(scale, bool) or not isinstance(scale, int) or not 0 <= scale <= 12 or rounding not in ROUNDINGS or len(inputs) != 2:
        return error("calculation requires two decimal strings, scale 0..12, and an allowed rounding mode")
    try:
        numbers = [Decimal(value) for value in inputs]
    except (InvalidOperation, TypeError, ValueError):
        return error("inputs must be decimal strings")
    if not all(number.is_finite() for number in numbers):
        return error("inputs must be finite")
    left, right = numbers
    try:
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
        else:
            raw = (right - left) / left * 100
    except (ZeroDivisionError, DecimalException):
        return error("invalid or out-of-range Decimal operation")
    try:
        quantized = raw.quantize(Decimal(1).scaleb(-scale), rounding=ROUNDINGS[rounding])
    except (InvalidOperation, ValueError, OverflowError):
        return error("result exceeds supported Decimal range")
    data = {"operation": operation, "inputs": list(inputs), "scale": scale, "rounding": rounding, "result": format(quantized, "f")}
    return result("ok", data)
