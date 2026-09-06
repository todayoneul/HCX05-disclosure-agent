from __future__ import annotations

import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from .common import result


_CURATED_COMMON_ALIASES = {
    "삼전": "삼성전자",
    "하닉": "sk하이닉스",
    "하이닉스": "sk하이닉스",
    "엔솔": "lg에너지솔루션",
    "엘지엔솔": "lg에너지솔루션",
    "기아자동차": "기아",
    "기아차": "기아",
    "포스코": "posco홀딩스",
    "포스코홀딩스": "posco홀딩스",
    "네이버": "naver",
    "삼성이앤에이": "삼성e&a",
    "엔씨": "nc",
    # Historical legal name of the in-universe successor.  The mapping is
    # installed only when the successor exists in the supplied universe.
    "삼성엔지니어링": "삼성e&a",
    "한화에어로": "한화에어로스페이스",
    "엘지씨엔에스": "lg씨엔에스",
    "엘지씨엔에스코퍼레이션": "lg씨엔에스",
}


# Korean phonetic reading of each Latin letter, used to accept a query that spells
# an initialism out in Hangul (e.g. "에스케이하이닉스" for "SK하이닉스").
_LETTER_TO_HANGUL = {
    "a": "에이", "b": "비", "c": "씨", "d": "디", "e": "이", "f": "에프",
    "g": "지", "h": "에이치", "i": "아이", "j": "제이", "k": "케이", "l": "엘",
    "m": "엠", "n": "엔", "o": "오", "p": "피", "q": "큐", "r": "알", "s": "에스",
    "t": "티", "u": "유", "v": "브이", "w": "더블유", "x": "엑스", "y": "와이",
    "z": "제트",
}
# Structural suffixes on English legal names, stripped so "Samsung Electronics"
# matches "SAMSUNG ELECTRONICS CO,.LTD".
_ENGLISH_NAME_SUFFIXES = (
    "corporation", "incorporated", "incorporation", "company", "limited",
    "holdings", "group", "coltd", "corp", "inc", "ltd", "co", "plc",
)


def _strip_english_suffixes(normalized: str) -> str:
    changed = True
    while changed:
        changed = False
        for suffix in _ENGLISH_NAME_SUFFIXES:
            if normalized.endswith(suffix) and len(normalized) - len(suffix) >= 2:
                normalized = normalized[: -len(suffix)]
                changed = True
    return normalized


def _hangul_letter_reading(name: str) -> str:
    """Spell out short Latin-letter initialisms (<=3 letters) in Hangul, leaving
    longer runs (real words like POSCO/NAVER) and Korean text untouched."""
    def replace(match: "re.Match[str]") -> str:
        run = match.group(0)
        if len(run) > 3:
            return run
        return "".join(_LETTER_TO_HANGUL.get(ch.casefold(), ch) for ch in run)

    return re.sub(r"[A-Za-z]+", replace, name)


def _normalize(value: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKC", value).casefold() if ch.isalnum())


# Trailing Korean particles (조사), longest first, so a query token like
# "삼성전자의" or "SK하이닉스를" reduces to the bare company name while a genuinely
# different name such as "카카오게임즈" (no particle to strip) does not collapse to
# a shorter in-universe company.
_TRAILING_JOSA = (
    "으로써", "으로서", "이라는", "이라고", "이라", "이란", "이나",
    "라는", "라고", "으로", "에서", "에게", "에도", "에는", "와의", "과의",
    "까지", "부터", "보다", "처럼", "만큼", "께서",
    "의", "은", "는", "이", "가", "을", "를", "와", "과", "에", "로", "도",
    "만", "라", "나", "란", "및",
)


def _strip_trailing_josa(token: str) -> str:
    for josa in _TRAILING_JOSA:
        if token.endswith(josa) and len(token) > len(josa):
            return token[: -len(josa)]
    return token


class CompanyResolver:
    def __init__(self, universe_csv: Path | str):
        with Path(universe_csv).open(encoding="utf-8-sig", newline="") as handle:
            self._rows = tuple(csv.DictReader(handle))
        aliases: dict[str, list[dict[str, str]]] = defaultdict(list)
        codes: dict[str, dict[str, str]] = {}
        def add_alias(key: str, row: dict[str, str]) -> None:
            if key and len(key) >= 2 and row not in aliases[key]:
                aliases[key].append(row)

        for row in self._rows:
            for field in ("corp_code", "stock_code"):
                codes[_normalize(row[field])] = row
            for field in ("corp_code", "stock_code", "corp_name", "listed_name", "corp_eng_name"):
                add_alias(_normalize(row.get(field, "")), row)
            # English legal name with structural suffixes removed
            # ("Samsung Electronics" from "SAMSUNG ELECTRONICS CO,.LTD").
            add_alias(_strip_english_suffixes(_normalize(row.get("corp_eng_name", ""))), row)
            # Hangul spelling of initialisms in the Korean/listed names
            # ("에스케이하이닉스" from "SK하이닉스", "엘지에너지솔루션" from "LG에너지솔루션").
            for field in ("corp_name", "listed_name"):
                add_alias(_normalize(_hangul_letter_reading(row.get(field, ""))), row)
            note = row.get("note", "")
            for match in re.finditer(r"(?:^|[,;/])\s*구\s+(.+?)(?=\s*\(|[,;/]|$)", note):
                normalized = _normalize(match.group(1))
                if normalized and row not in aliases[normalized]:
                    aliases[normalized].append(row)
        rows_by_name = {
            _normalize(row.get("corp_name", "")): row
            for row in self._rows
            if _normalize(row.get("corp_name", ""))
        }
        for shorthand, official_name in _CURATED_COMMON_ALIASES.items():
            row = rows_by_name.get(_normalize(official_name))
            normalized = _normalize(shorthand)
            if row is not None and row not in aliases[normalized]:
                aliases[normalized].append(row)
        self._aliases = aliases
        self._codes = codes

    @staticmethod
    def _sector_match_position(query: str, sector: str) -> int | None:
        """Return the first strong sector mention without prefix guessing.

        A component such as ``게임`` must not match the longer company token
        ``카카오게임즈``.  Component aliases are therefore accepted only when
        followed by an explicit group/conjunction boundary used by ranking
        questions.  The complete canonical label remains an exact compact match.
        """
        normalized_query = unicodedata.normalize("NFKC", query).casefold()
        components = tuple(
            value.strip()
            for value in re.split(r"[·ㆍ/&,+]", sector)
            if len(_normalize(value)) >= 2
        )
        boundary = (
            r"(?=\s*(?:와|과|및|·|ㆍ|/|&|,|"
            r"[0-9]+\s*사(?:\b|중)?|"
            r"기업|회사|업체|종목|섹터|산업|업종|군|중|내|$))"
        )
        positions: list[int] = []
        if components:
            canonical_pattern = r"[·ㆍ/&,+\s]*".join(
                re.escape(component.casefold()) for component in components
            )
            canonical_match = re.search(
                canonical_pattern + boundary, normalized_query
            )
            if canonical_match is not None:
                positions.append(canonical_match.start())
        for component in components:
            match = re.search(re.escape(component.casefold()) + boundary, normalized_query)
            if match is not None:
                positions.append(match.start())
        return min(positions) if positions else None

    def resolve_sector(self, query: str) -> dict:
        """Resolve one universe sector and return its complete candidate set.

        Sector membership is metadata from the supplied ``universe.csv`` only;
        it is never inferred from a company's prose or from a model response.
        """
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            return result(
                "error",
                {},
                limitations=["sector query must contain 1..1000 characters"],
            )
        sectors: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in self._rows:
            sector = row.get("sector", "").strip()
            if sector:
                sectors[sector].append(row)
        matches = sorted(
            (
                (position, sector)
                for sector in sectors
                if (position := self._sector_match_position(query, sector)) is not None
            ),
            key=lambda value: (value[0], value[1]),
        )
        if not matches:
            return result(
                "not_found",
                [],
                limitations=["sector is outside the supplied universe"],
            )
        if len(matches) > 1:
            return result(
                "ambiguous",
                [{"sector": sector} for _, sector in matches],
                limitations=["query names multiple supplied sectors"],
            )
        sector = matches[0][1]
        projected = [
            {
                key: row.get(key, "")
                for key in (
                    "corp_code",
                    "stock_code",
                    "corp_name",
                    "listed_name",
                    "sector",
                )
            }
            for row in sectors[sector]
        ]
        return result(
            "ok",
            {"sector": sector, "candidates": projected},
        )

    def resolve_company(self, query: str) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            return result("error", {}, limitations=["company query must contain 1..1000 characters"])
        normalized = _normalize(query)
        candidates = [self._codes[normalized]] if normalized in self._codes else sorted(self._aliases.get(normalized, []), key=lambda row: row["corp_code"])
        if not candidates and not re.search(r";|--|/\*|\*/", query):
            # Match runs of consecutive query tokens (letters kept together, digits
            # split off) against known aliases, tolerating one trailing particle and
            # English legal suffixes. Joining consecutive tokens lets a multi-word
            # name resolve inside a question ("LG CNS Co., Ltd.의 매출"), while a
            # longer distinct name that is itself a single token and merely starts
            # with an in-universe name ("카카오게임즈") never splits and stays unknown.
            tokens = re.findall(
                r"[A-Za-z가-힣&]+|[0-9]+", unicodedata.normalize("NFKC", query)
            )
            matched: dict[str, dict[str, str]] = {}
            for start in range(len(tokens)):
                for end in range(start, min(start + 5, len(tokens))):
                    joined = "".join(tokens[start : end + 1])
                    keys = set()
                    for form in (joined, _strip_trailing_josa(joined)):
                        normalized_form = _normalize(form)
                        keys.add(normalized_form)
                        keys.add(_strip_english_suffixes(normalized_form))
                    for key in keys:
                        if len(key) >= 2:
                            for row in self._aliases.get(key, []):
                                matched.setdefault(row["corp_code"], row)
            candidates = [matched[code] for code in sorted(matched)]
        if not candidates and not re.search(r";|--|/\*|\*/", query):
            # No-space input can collapse several names and the rest of a clause
            # into one run ("CompareSamsungElectronics와에스케이하이닉스의…").
            # Find every known alias only at a strong semantic boundary: a Korean
            # particle/conjunction, an English comparison conjunction, or a DART
            # document/basis marker.  This recovers each company in a comparison
            # while a longer unrelated name ("카카오게임즈", "Nokia") still cannot
            # collapse to the shorter in-universe 카카오/Kia alias.
            runs = re.findall(
                r"[A-Za-z가-힣0-9&]+", unicodedata.normalize("NFKC", query)
            )
            right_boundaries = tuple(
                dict.fromkeys(
                    (
                        *(_normalize(value) for value in _TRAILING_JOSA),
                        "사업보고서",
                        "분기보고서",
                        "반기보고서",
                        "공시",
                        "연결",
                        "별도",
                        "and",
                        "versus",
                        "vs",
                        # Strong question-intent markers.  These recover a bare
                        # no-space name such as ``삼성전자매출액`` without
                        # treating an arbitrary longer word/company name as a
                        # prefix match (``카카오게임즈`` still has no boundary).
                        "매출액",
                        "매출",
                        "영업수익",
                        "영업이익",
                        "영업손실",
                        "당기순이익",
                        "당기순손실",
                        "순이익",
                        "순손실",
                        "자산총계",
                        "부채총계",
                        "자본총계",
                        "배당",
                        "최대주주",
                        "직원수",
                        "종업원수",
                        "설립일",
                        "본점주소",
                        "회사개요",
                        "사업내용",
                        "주요사업",
                        "주요제품",
                        "시설투자",
                        "설비투자",
                        "자금조달",
                        "revenue",
                        "sales",
                        "operatingprofit",
                        "operatingloss",
                        "netincome",
                        "netloss",
                        "totalassets",
                        "totalliabilities",
                        "totalequity",
                        "dividend",
                        "eps",
                        "roe",
                    )
                )
            )
            left_boundaries = tuple(
                dict.fromkeys(
                    (
                        *(_normalize(value) for value in _TRAILING_JOSA),
                        "제출된",
                        "공시된",
                        "and",
                        "compare",
                        "summarize",
                        "summary",
                        "explain",
                        "report",
                        "show",
                        "tell",
                        "analyze",
                        "versus",
                        "vs",
                        "년",
                        "월",
                        "일",
                        "분기",
                        "기준",
                        "please",
                    )
                )
            )

            def script_group(character: str) -> str:
                if "가" <= character <= "힣":
                    return "hangul"
                if character.isascii() and character.isalpha():
                    return "latin"
                if character.isdigit():
                    return "digit"
                return "other"

            matched: dict[str, dict[str, str]] = {}
            for run in runs:
                normalized_run = _normalize(run)
                for alias, rows in self._aliases.items():
                    if len(alias) < 2 or len(alias) > len(normalized_run):
                        continue
                    start = normalized_run.find(alias)
                    while start >= 0:
                        end = start + len(alias)
                        left = normalized_run[:start]
                        right = normalized_run[end:]
                        left_ok = (
                            not left
                            or any(left.endswith(value) for value in left_boundaries)
                            or script_group(left[-1]) != script_group(alias[0])
                        )
                        right_ok = (
                            not right
                            or any(right.startswith(value) for value in right_boundaries)
                        )
                        if left_ok and right_ok:
                            for row in rows:
                                matched.setdefault(row["corp_code"], row)
                        start = normalized_run.find(alias, start + 1)
            candidates = [matched[code] for code in sorted(matched)]
        projected = [
            {key: row.get(key, "") for key in ("corp_code", "stock_code", "corp_name", "listed_name", "sector")}
            for row in candidates
        ]
        if not projected:
            return result("not_found", [], limitations=["company is outside the supplied universe"])
        if len(projected) > 1:
            return result("ambiguous", projected, limitations=["normalized alias matches multiple supplied companies"])
        return result("ok", projected[0])
