"""Closed HCX tool schemas and a model-independent safe dispatcher."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

from disclosure_agent.context import (
    EvidenceItem,
    evidence_from_search_result,
    evidence_from_section_chunk,
)
from disclosure_agent.tools.calculate import calculate


_TOOL_NAMES = (
    "resolve_company",
    "query_events",
    "list_filings",
    "list_sections",
    "read_section",
    "search_chunks",
    "get_history",
    "calculate",
)
_GROUNDED_TOOLS = frozenset(_TOOL_NAMES) - {"resolve_company", "calculate"}
_CANONICAL_CITATION_KEYS = frozenset(
    (
        "doc_id",
        "rcept_no",
        "corp_code",
        "corp_name",
        "report_nm",
        "rcept_dt",
        "section",
        "is_latest",
        "root_rcept_no",
        "latest_rcept_no",
        "correction_status",
        "correction_method",
    )
)
_RESULT_STATUSES = frozenset(("ok", "not_found", "ambiguous", "info_limit", "error"))
_DECIMAL_PATTERN = re.compile(r"^-?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?$")
_DEFAULT_MAX_RESULT_CHARS = 65_536
_MAX_JSON_DEPTH = 32
_DEFAULT_ARGUMENTS: dict[str, dict[str, Any]] = {
    "query_events": {"latest_only": True, "limit": 50},
    "list_filings": {"latest_only": True, "limit": 50},
    "list_sections": {"limit": 50},
    "read_section": {"max_chars": 12_000},
    "search_chunks": {"latest_only": True, "k": 10},
    "calculate": {"scale": 2, "rounding": "ROUND_HALF_UP"},
}


class ToolRegistryConfigurationError(ValueError):
    """Raised when immutable tool backends cannot be safely bound."""


class _LineageChangedError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolLineage:
    pipeline_release: str
    retrieval_release: str


@dataclass(frozen=True)
class ToolDispatchError:
    code: str
    message: str


@dataclass(frozen=True)
class ToolDispatchResult:
    tool_name: str
    status: str
    data: Any
    citations: tuple[Mapping[str, Any], ...]
    limitations: tuple[str, ...]
    evidence: tuple[EvidenceItem, ...]
    error: ToolDispatchError | None
    lineage: ToolLineage

    def to_model_payload(self) -> dict[str, Any]:
        """Return a detached JSON-safe payload suitable for one tool message."""
        return {
            "tool": self.tool_name,
            "status": self.status,
            "data": _thaw_json(self.data),
            "citations": [_thaw_json(item) for item in self.citations],
            "limitations": list(self.limitations),
            "error": (
                None
                if self.error is None
                else {"code": self.error.code, "message": self.error.message}
            ),
            "lineage": {
                "pipeline_release": self.lineage.pipeline_release,
                "retrieval_release": self.lineage.retrieval_release,
            },
        }


def _string(**constraints: Any) -> dict[str, Any]:
    return {"type": "string", **constraints}


def _integer(minimum: int, maximum: int) -> dict[str, Any]:
    return {"type": "integer", "minimum": minimum, "maximum": maximum}


def _array(items: dict[str, Any], minimum: int, maximum: int) -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": minimum,
        "maxItems": maximum,
        "items": items,
    }


def _parameters(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


_CORP_CODE = _string(minLength=1, maxLength=8, pattern="^[0-9]+$")
_CORP_NAME = _string(minLength=1, maxLength=200)
_DATE = _string(pattern="^[0-9]{8}$")
_DOC_ID = _string(minLength=1, maxLength=100)
_RCEPT_NO = _string(pattern="^[0-9]{14}$")
_DOC_SUBTYPE = _string(minLength=1, maxLength=100)
_DECIMAL = _string(minLength=1, maxLength=100)

_SCHEMAS = (
    _tool(
        "resolve_company",
        "Resolve one supplied-corpus company name or code.",
        _parameters({"query": _string(minLength=1, maxLength=200)}, ["query"]),
    ),
    _tool(
        "query_events",
        "Query structured disclosure events for one company by corp_code or corp_name.",
        _parameters(
            {
                "corp_code": _CORP_CODE,
                "corp_name": _CORP_NAME,
                "event_types": _array(
                    _string(minLength=1, maxLength=100), 1, 20
                ),
                "rcept_from": _DATE,
                "rcept_to": _DATE,
                "event_from": _DATE,
                "event_to": _DATE,
                "amount_min": _DECIMAL,
                "amount_max": _DECIMAL,
                "latest_only": {"type": "boolean"},
                "limit": _integer(1, 50),
            },
            [],
        ),
    ),
    _tool(
        "list_filings",
        "List bounded disclosure metadata for one company by corp_code or corp_name.",
        _parameters(
            {
                "corp_code": _CORP_CODE,
                "corp_name": _CORP_NAME,
                "doc_group": _string(
                    enum=["periodic", "exchange", "major", "holding"]
                ),
                "doc_subtype": _DOC_SUBTYPE,
                "base_year": _integer(1900, 9999),
                "base_month": _integer(1, 12),
                "rcept_from": _DATE,
                "rcept_to": _DATE,
                "latest_only": {"type": "boolean"},
                "limit": _integer(1, 50),
            },
            [],
        ),
    ),
    _tool(
        "list_sections",
        "List section paths for exactly one filing identifier.",
        _parameters(
            {"doc_id": _DOC_ID, "rcept_no": _RCEPT_NO, "limit": _integer(1, 50)},
            [],
        ),
    ),
    _tool(
        "read_section",
        "Read a bounded exact section from exactly one filing.",
        _parameters(
            {
                "path": _string(minLength=1, maxLength=1000),
                "doc_id": _DOC_ID,
                "rcept_no": _RCEPT_NO,
                "max_chars": _integer(1, 12_000),
            },
            ["path"],
        ),
    ),
    _tool(
        "search_chunks",
        "Search verified filing chunks with bounded lexical retrieval.",
        _parameters(
            {
                "query": _string(minLength=1, maxLength=1000),
                "corp_code": _CORP_CODE,
                "doc_subtype": _DOC_SUBTYPE,
                "base_year": _integer(1900, 9999),
                "latest_only": {"type": "boolean"},
                "path_hint": _string(minLength=1, maxLength=500),
                "k": _integer(1, 20),
            },
            ["query"],
        ),
    ),
    _tool(
        "get_history",
        "Get the correction chain for exactly one filing identifier.",
        _parameters({"doc_id": _DOC_ID, "rcept_no": _RCEPT_NO}, []),
    ),
    _tool(
        "calculate",
        "Run one bounded Decimal calculation on exactly two decimal strings.",
        _parameters(
            {
                "operation": _string(
                    enum=[
                        "add",
                        "subtract",
                        "multiply",
                        "divide",
                        "ratio_percent",
                        "percent_change",
                    ]
                ),
                "inputs": _array(_DECIMAL, 2, 2),
                "scale": _integer(0, 12),
                "rounding": _string(
                    enum=[
                        "ROUND_HALF_UP",
                        "ROUND_HALF_EVEN",
                        "ROUND_DOWN",
                        "ROUND_UP",
                    ]
                ),
            },
            ["operation", "inputs"],
        ),
    ),
)


def _freeze_json(
    value: Any,
    label: str = "value",
    *,
    _seen: set[int] | None = None,
    _depth: int = 0,
) -> Any:
    if _depth > _MAX_JSON_DEPTH:
        raise ValueError(f"{label} exceeds the JSON depth bound")
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite JSON")
        return value
    if isinstance(value, Mapping):
        seen = set() if _seen is None else _seen
        identity = id(value)
        if identity in seen:
            raise ValueError(f"{label} contains a JSON cycle")
        seen.add(identity)
        try:
            frozen: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"{label} keys must be strings")
                frozen[key] = _freeze_json(
                    item, f"{label}.{key}", _seen=seen, _depth=_depth + 1
                )
            return MappingProxyType(frozen)
        finally:
            seen.remove(identity)
    if isinstance(value, (list, tuple)):
        seen = set() if _seen is None else _seen
        identity = id(value)
        if identity in seen:
            raise ValueError(f"{label} contains a JSON cycle")
        seen.add(identity)
        try:
            return tuple(
                _freeze_json(item, f"{label}[]", _seen=seen, _depth=_depth + 1)
                for item in value
            )
        finally:
            seen.remove(identity)
    raise ValueError(f"{label} must contain only JSON values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _validate_schema_value(value: Any, schema: Mapping[str, Any], label: str) -> None:
    expected = schema["type"]
    if expected == "string":
        if type(value) is not str:
            raise ValueError(f"{label} must be a string")
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get(
            "maxLength", len(value)
        ):
            raise ValueError(f"{label} length is outside the allowed range")
        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(f"{label} is outside the allowed enum")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise ValueError(f"{label} does not match the allowed format")
        if any(ord(character) < 32 for character in value):
            raise ValueError(f"{label} contains control characters")
        return
    if expected == "integer":
        if type(value) is not int:
            raise ValueError(f"{label} must be an integer")
        if not schema["minimum"] <= value <= schema["maximum"]:
            raise ValueError(f"{label} is outside the allowed range")
        return
    if expected == "boolean":
        if type(value) is not bool:
            raise ValueError(f"{label} must be a boolean")
        return
    if expected == "array":
        if type(value) is not list:
            raise ValueError(f"{label} must be an array")
        if not schema["minItems"] <= len(value) <= schema["maxItems"]:
            raise ValueError(f"{label} length is outside the allowed range")
        for index, item in enumerate(value):
            _validate_schema_value(item, schema["items"], f"{label}[{index}]")
        return
    raise ValueError(f"{label} has an unsupported schema")


def _validate_date(value: str, label: str) -> None:
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        raise ValueError(f"{label} must be a valid calendar date") from None


def _validate_decimal(value: str, label: str) -> None:
    if _DECIMAL_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a decimal string")
    try:
        parsed = Decimal(value.replace(",", ""))
    except InvalidOperation:
        raise ValueError(f"{label} must be a decimal string") from None
    if not parsed.is_finite():
        raise ValueError(f"{label} must be finite")


def _validate_semantics(tool_name: str, arguments: Mapping[str, Any]) -> None:
    if tool_name in {"query_events", "list_filings"}:
        if "corp_code" not in arguments and "corp_name" not in arguments:
            raise ValueError("corp_code or corp_name is required")
    if tool_name in {"list_sections", "read_section", "get_history"}:
        if ("doc_id" in arguments) == ("rcept_no" in arguments):
            raise ValueError("exactly one doc_id or rcept_no is required")
    if "doc_id" in arguments:
        doc_id = arguments["doc_id"]
        if ".." in doc_id or "/" in doc_id or "\\" in doc_id:
            raise ValueError("doc_id must not be a path")
    if tool_name == "read_section":
        section_path = arguments["path"]
        if (
            section_path.startswith("/")
            or "\\" in section_path
            or re.search(r"(^|/)\.\.(/|$)", section_path)
            or re.match(r"^[A-Za-z]:", section_path)
        ):
            raise ValueError("section path must not be a filesystem path")
    for label in ("rcept_from", "rcept_to", "event_from", "event_to"):
        if label in arguments:
            _validate_date(arguments[label], label)
    for start, end in (("rcept_from", "rcept_to"), ("event_from", "event_to")):
        if start in arguments and end in arguments and arguments[start] > arguments[end]:
            raise ValueError(f"{start} must not exceed {end}")
    for label in ("amount_min", "amount_max"):
        if label in arguments:
            _validate_decimal(arguments[label], label)
    if "amount_min" in arguments and "amount_max" in arguments:
        low = Decimal(arguments["amount_min"].replace(",", ""))
        high = Decimal(arguments["amount_max"].replace(",", ""))
        if low > high:
            raise ValueError("amount_min must not exceed amount_max")
    if tool_name == "calculate":
        for index, value in enumerate(arguments["inputs"]):
            _validate_decimal(value, f"inputs[{index}]")


def _validate_citation(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CANONICAL_CITATION_KEYS:
        raise ValueError("citation must contain exactly the canonical keys")
    for key, item in value.items():
        if key == "is_latest":
            if type(item) is not bool:
                raise ValueError("citation is_latest must be boolean")
        elif type(item) is not str:
            raise ValueError(f"citation {key} must be a string")
    return _freeze_json(value, "citation")


def _citation_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in value}


def _embedded_citations(
    tool_name: str, data: Any, status: str
) -> list[Mapping[str, Any]]:
    if status != "ok" and not data:
        return []
    if tool_name in {"query_events", "list_filings", "list_sections", "search_chunks"}:
        if not isinstance(data, list):
            raise ValueError("grounded list tool data must be a list")
        rows = data
    elif tool_name == "read_section":
        if not isinstance(data, Mapping) or not isinstance(data.get("chunks"), list):
            raise ValueError("section data must contain chunks")
        rows = data["chunks"]
    elif tool_name == "get_history":
        if not isinstance(data, Mapping) or not isinstance(data.get("chain"), list):
            raise ValueError("history data must contain a chain")
        rows = data["chain"]
        queried = data.get("queried_correction")
        if queried is not None:
            if not isinstance(queried, Mapping) or not isinstance(
                queried.get("citation"), Mapping
            ):
                raise ValueError("queried correction citation differs")
            if not any(queried["citation"] == row.get("citation") for row in rows):
                raise ValueError("queried correction citation is outside history chain")
    else:
        return []
    embedded: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("citation"), Mapping):
            raise ValueError("grounded row is missing a citation")
        embedded.append(row["citation"])
    return embedded


def _canonicalize_structured_sections(
    tool_name: str,
    data: Any,
    citations: list[Mapping[str, Any]],
    status: str,
) -> None:
    label = {
        "list_filings": "filing_metadata",
        "get_history": "correction_history",
    }.get(tool_name)
    if label is None or status != "ok":
        return
    embedded = _embedded_citations(tool_name, data, status)
    for citation in [*citations, *embedded]:
        if not citation["section"]:
            citation["section"] = label
    if tool_name == "get_history" and data.get("queried_correction") is not None:
        queried_citation = data["queried_correction"]["citation"]
        if not queried_citation["section"]:
            queried_citation["section"] = label


def _structured_rows(tool_name: str, data: Any) -> list[Mapping[str, Any]]:
    if tool_name in {"query_events", "list_filings", "list_sections"}:
        return list(data)
    if tool_name == "get_history":
        rows = [dict(row) for row in data["chain"]]
        queried = data.get("queried_correction")
        if queried is not None:
            target = queried["citation"]["rcept_no"]
            for row in rows:
                if row["citation"]["rcept_no"] == target:
                    row["queried_correction"] = {
                        key: value for key, value in queried.items() if key != "citation"
                    }
                    break
        return rows
    return []


def _structured_evidence(tool_name: str, data: Any) -> tuple[EvidenceItem, ...]:
    evidence: list[EvidenceItem] = []
    for rank, row in enumerate(_structured_rows(tool_name, data), 1):
        citation = _citation_dict(row["citation"])
        body = {key: value for key, value in row.items() if key != "citation"}
        text = json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        evidence.append(
            EvidenceItem(
                source_id=f"{tool_name}:{citation['doc_id']}:{digest}",
                text=text,
                citation=citation,
                source_kind=tool_name,
                priority=1,
                rank=rank,
            )
        )
    return tuple(evidence)


class ToolRegistry:
    """Dispatch the fixed tool surface against already-bound immutable snapshots."""

    def __init__(
        self,
        disclosure_tools: Any,
        retrieval_index: Any,
        *,
        max_result_chars: int = _DEFAULT_MAX_RESULT_CHARS,
    ) -> None:
        if type(max_result_chars) is not int or not 256 <= max_result_chars <= 1_000_000:
            raise ToolRegistryConfigurationError(
                "max_result_chars must be an integer within 256..1000000"
            )
        try:
            disclosure_release = Path(disclosure_tools.release).resolve()
            retrieval_pipeline_release = Path(retrieval_index.pipeline_release).resolve()
            retrieval_release = Path(retrieval_index.release).resolve()
        except (AttributeError, TypeError, ValueError, OSError):
            raise ToolRegistryConfigurationError(
                "tool backends must expose bound release paths"
            ) from None
        if disclosure_release != retrieval_pipeline_release:
            raise ToolRegistryConfigurationError("tool backends have different pipeline lineage")
        self._disclosure = disclosure_tools
        self._retrieval = retrieval_index
        self._max_result_chars = max_result_chars
        self._bound_paths: tuple[tuple[Any, str, Path], ...] = tuple(
            item
            for item in (
                (disclosure_tools, "release", disclosure_release),
                (
                    disclosure_tools,
                    "db_path",
                    Path(disclosure_tools.db_path).resolve(),
                )
                if hasattr(disclosure_tools, "db_path")
                else None,
                (retrieval_index, "pipeline_release", retrieval_pipeline_release),
                (retrieval_index, "release", retrieval_release),
            )
            if item is not None
        )
        self.lineage = ToolLineage(
            pipeline_release=disclosure_release.name,
            retrieval_release=retrieval_release.name,
        )
        self.schemas = tuple(_freeze_json(schema, "tool schema") for schema in _SCHEMAS)
        self._parameters = {
            schema["function"]["name"]: schema["function"]["parameters"]
            for schema in _SCHEMAS
        }

    def schema_payload(self) -> list[dict[str, Any]]:
        """Return a detached schema copy without exposing registry aliases."""
        return [_thaw_json(schema) for schema in self.schemas]

    def dispatch(self, tool_name: str, arguments: Mapping[str, Any]) -> ToolDispatchResult:
        if tool_name not in _TOOL_NAMES:
            return self._error(tool_name if isinstance(tool_name, str) else "", "unknown_tool")
        try:
            detached = self._validated_arguments(tool_name, arguments)
        except (TypeError, ValueError):
            return self._error(tool_name, "invalid_arguments")
        if tool_name in {"query_events", "list_filings"}:
            resolved = self._resolve_company_argument(tool_name, detached)
            if resolved is not None:
                return resolved
        try:
            self._assert_lineage()
            response = self._execute(tool_name, detached)
            self._assert_lineage()
        except _LineageChangedError:
            return self._error(tool_name, "lineage_changed")
        except Exception:
            return self._error(tool_name, "tool_execution_failed")
        return self._normalize(tool_name, response)

    def _resolve_company_argument(
        self, tool_name: str, detached: dict[str, Any]
    ) -> ToolDispatchResult | None:
        """Resolve a supplied ``corp_name`` into ``corp_code`` in place.

        Returns ``None`` when a canonical ``corp_code`` is available to execute
        with; otherwise returns a terminal not_found/ambiguous/error result so a
        natural-language company reference never falls through to a hard error.
        """
        corp_name = detached.pop("corp_name", None)
        if "corp_code" in detached:
            return None
        try:
            self._assert_lineage()
            resolution = self._disclosure.resolve_company(corp_name)
            self._assert_lineage()
        except _LineageChangedError:
            return self._error(tool_name, "lineage_changed")
        except Exception:
            return self._error(tool_name, "tool_execution_failed")
        status = resolution.get("status") if isinstance(resolution, Mapping) else None
        if status != "ok":
            reported = status if status in {"not_found", "ambiguous"} else "not_found"
            return self._normalize(
                tool_name,
                {
                    "status": reported,
                    "data": [],
                    "citations": [],
                    "limitations": ["named company was not uniquely resolved"],
                },
            )
        corp_code = resolution["data"].get("corp_code") if isinstance(resolution["data"], Mapping) else None
        if not isinstance(corp_code, str) or not corp_code:
            return self._error(tool_name, "tool_execution_failed")
        detached["corp_code"] = corp_code
        return None

    def _assert_lineage(self) -> None:
        for backend, attribute, expected in self._bound_paths:
            try:
                current = Path(getattr(backend, attribute)).resolve()
            except (AttributeError, TypeError, ValueError, OSError):
                raise _LineageChangedError from None
            if current != expected:
                raise _LineageChangedError

    def _validated_arguments(
        self, tool_name: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(arguments, Mapping):
            raise TypeError("arguments must be an object")
        parameters = self._parameters[tool_name]
        properties = parameters["properties"]
        if not all(isinstance(key, str) for key in arguments):
            raise ValueError("argument keys must be strings")
        if not set(arguments).issubset(properties):
            raise ValueError("unknown argument")
        if not set(parameters["required"]).issubset(arguments):
            raise ValueError("required argument is missing")
        effective = dict(_DEFAULT_ARGUMENTS.get(tool_name, {}))
        effective.update(arguments)
        for key, value in effective.items():
            _validate_schema_value(value, properties[key], key)
        _validate_semantics(tool_name, effective)
        return _thaw_json(_freeze_json(effective, "arguments"))

    def _execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name == "resolve_company":
            return self._disclosure.resolve_company(arguments["query"])
        if tool_name == "query_events":
            values = dict(arguments)
            corp_code = values.pop("corp_code")
            return self._disclosure.query_events(corp_code, **values)
        if tool_name == "list_filings":
            values = dict(arguments)
            corp_code = values.pop("corp_code")
            return self._disclosure.list_filings(corp_code, **values)
        if tool_name in {"list_sections", "read_section", "get_history"}:
            return getattr(self._disclosure, tool_name)(**arguments)
        if tool_name == "search_chunks":
            values = dict(arguments)
            query = values.pop("query")
            return self._retrieval.search_chunks(query, **values)
        if tool_name == "calculate":
            values = dict(arguments)
            operation = values.pop("operation")
            inputs = values.pop("inputs")
            return calculate(operation, inputs, **values)
        raise AssertionError("closed registry route is missing")

    def _normalize(self, tool_name: str, response: Any) -> ToolDispatchResult:
        try:
            if not isinstance(response, Mapping) or not {
                "status",
                "data",
                "citations",
                "limitations",
            }.issubset(response):
                raise ValueError("tool result schema differs")
            status = response["status"]
            if status not in _RESULT_STATUSES:
                raise ValueError("tool result status differs")
            if status == "error":
                return self._error(tool_name, "tool_rejected_arguments")
            limitations = response["limitations"]
            citations = response["citations"]
            if not isinstance(limitations, list) or not all(
                type(item) is str for item in limitations
            ):
                raise ValueError("tool limitations differ")
            if not isinstance(citations, list):
                raise ValueError("tool citations differ")
            normalized_data = _thaw_json(_freeze_json(response["data"], "tool data"))
            normalized_citations = _thaw_json(
                _freeze_json(citations, "tool citations")
            )
            _canonicalize_structured_sections(
                tool_name, normalized_data, normalized_citations, status
            )
            frozen_citations = tuple(
                _validate_citation(item) for item in normalized_citations
            )
            embedded = tuple(
                _validate_citation(item)
                for item in _embedded_citations(tool_name, normalized_data, status)
            )
            if embedded != frozen_citations:
                raise ValueError("outer and embedded citations differ")
            data = _freeze_json(normalized_data, "tool data")
            detached_core = {
                "status": status,
                "data": _thaw_json(data),
                "citations": [_thaw_json(item) for item in frozen_citations],
                "limitations": list(limitations),
            }
            rendered = json.dumps(
                detached_core,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            if len(rendered) > self._max_result_chars:
                return self._error(tool_name, "result_too_large")
            if tool_name in _GROUNDED_TOOLS and status == "ok" and bool(normalized_data):
                if not frozen_citations:
                    raise ValueError("grounded tool result is missing citations")
            evidence = self._evidence(tool_name, normalized_data, status)
        except Exception:
            return self._error(tool_name, "malformed_tool_result")
        return ToolDispatchResult(
            tool_name=tool_name,
            status=status,
            data=data,
            citations=frozen_citations,
            limitations=tuple(limitations),
            evidence=evidence,
            error=None,
            lineage=self.lineage,
        )

    def _evidence(self, tool_name: str, data: Any, status: str) -> tuple[EvidenceItem, ...]:
        if status != "ok":
            return ()
        if tool_name == "search_chunks":
            if not isinstance(data, list):
                raise ValueError("search data must be a list")
            return tuple(
                evidence_from_search_result(row, rank=rank)
                for rank, row in enumerate(data, 1)
            )
        if tool_name == "read_section":
            if not isinstance(data, Mapping) or not isinstance(data.get("chunks"), list):
                raise ValueError("section data must contain chunks")
            return tuple(
                evidence_from_section_chunk(row, rank=rank)
                for rank, row in enumerate(data["chunks"], 1)
            )
        if tool_name in {"query_events", "list_filings", "list_sections", "get_history"}:
            return _structured_evidence(tool_name, data)
        return ()

    def _error(self, tool_name: str, code: str) -> ToolDispatchResult:
        messages = {
            "unknown_tool": "The requested tool is not available.",
            "invalid_arguments": "The tool arguments violate the closed contract.",
            "tool_execution_failed": "The tool could not complete safely.",
            "tool_rejected_arguments": "The tool rejected the validated request.",
            "malformed_tool_result": "The tool returned an invalid result.",
            "result_too_large": "The tool result exceeds the bounded response size.",
            "lineage_changed": "The bound tool snapshot changed during dispatch.",
        }
        return ToolDispatchResult(
            tool_name=tool_name,
            status="error",
            data=MappingProxyType({}),
            citations=(),
            limitations=(),
            evidence=(),
            error=ToolDispatchError(code=code, message=messages[code]),
            lineage=self.lineage,
        )


__all__ = [
    "ToolDispatchError",
    "ToolDispatchResult",
    "ToolLineage",
    "ToolRegistry",
    "ToolRegistryConfigurationError",
]
