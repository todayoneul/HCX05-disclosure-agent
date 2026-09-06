"""Conservative deterministic extraction of company profile facts.

Extracts founding_date, headquarters, and ceo from source-ordered section triples
(section, citationMapping, text) of ONE verified filing.
No I/O, model calls, or runner imports.

Caller/runner is responsible for subject locking, period binding, and passing
complete, verified section triples. This module strictly enforces within-filing
consistency, anchored scope (parent vs subsidiary/former), and fails closed on
unsupported, unverified, ambiguous, or conflicting evidence.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import re
from types import MappingProxyType


Group = tuple[str, Mapping[str, object], str]

_DATE_YMD = re.compile(r"([0-9]{4}년\s*[0-9]{1,2}월\s*[0-9]{1,2}일)")
_DATE_DOT = re.compile(r"([0-9]{4}\.[0-9]{2}\.[0-9]{2})")
_FORMER = re.compile(r"前|전임|퇴임|퇴직|사임|고문|자문|상담역|(?:^|\s)전\s*대표|former|retired", re.I)
_CEO_TERM = re.compile(r"대표\s*이사|(?:^|\s)CEO(?:\s|$)")

_EXCLUDED_ADDR_PREFIX = re.compile(r"자회사|종속회사|계열사|공장|지점|영업소|홈페이지|이메일|인터넷")
_GENERIC_ABSENCE = frozenset({"-", "해당없음", "해당사항 없음", "해당사항없음", "미정", "미상", "없음", ""})


@dataclass(frozen=True)
class ProfileFact:
    kind: str
    value: str
    citation: Mapping[str, object]
    limitations: tuple[str, ...] = ()

    @property
    def field(self) -> str:
        return self.kind


CompanyProfileFact = ProfileFact


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _cell(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<br\s*/?>", " ", value, flags=re.I)).strip()


def _valid_group(group: Group) -> bool:
    if not isinstance(group, (tuple, list)) or len(group) != 3:
        return False
    section, c, text = group
    keys = ("corp_code", "corp_name", "report_nm", "rcept_no", "root_rcept_no", "latest_rcept_no", "section")
    return (
        isinstance(text, str)
        and isinstance(section, str)
        and isinstance(c, Mapping)
        and all(isinstance(c.get(k), str) and c[k].strip() for k in keys)
        and c["section"] == section
        and all(re.fullmatch(r"[0-9]{14}", str(c[k])) for k in ("rcept_no", "root_rcept_no", "latest_rcept_no"))
        and c.get("is_latest") is True
        and c["latest_rcept_no"] == c["rcept_no"]
        and c.get("correction_status") in {"original", "linked"}
        and not c.get("truncated")
        and not c.get("remaining_parts")
    )


def _company_aliases(corp_name: str, texts: Sequence[str]) -> set[str]:
    aliases = {corp_name, _compact(corp_name)}
    known_pairs = {
        "케이티": {"KT", "케이티"},
        "KT": {"KT", "케이티"},
        "HMM": {"HMM", "에이치엠엠"},
        "에이치엠엠": {"HMM", "에이치엠엠"},
        "현대글로비스": {"현대글로비스", "글로비스"},
        "에코프로비엠": {"에코프로비엠", "에코프로BM"},
        "하이브": {"하이브", "빅히트"},
    }
    if corp_name in known_pairs:
        aliases.update(known_pairs[corp_name])
    for text in texts:
        m = re.search(r"명칭은\s*['\"『]([^'\"』]+)['\"』]", text)
        if m:
            clean = m.group(1).replace("주식회사", "").replace("(주)", "").replace("㈜", "").strip()
            if clean:
                aliases.add(clean)
        m_eng = re.search(r"영문명(?:은)?\s*['\"『]([^'\"』]+)['\"』]", text)
        if m_eng:
            clean_eng = re.sub(r",?\s*(?:Inc\.|Co\.,?\s*Ltd\.|Corporation|Company Limited).*", "", m_eng.group(1), flags=re.I).strip()
            if clean_eng:
                aliases.add(clean_eng)
    return {a for a in aliases if len(a) >= 2}


def _verify_question_company(question: str, aliases: set[str]) -> bool:
    known_other_corps = {
        "삼성전자", "현대자동차", "SK하이닉스", "LG에너지솔루션", "삼성바이오로직스",
        "포스코홀딩스", "기아", "LG화학", "삼성SDI", "카카오", "NAVER", "네이버",
    }
    if any(other in question for other in known_other_corps if other not in aliases):
        return False
    return True


def _valid_date_str(val: str) -> str | None:
    found = _DATE_YMD.search(val) or _DATE_DOT.search(val)
    if found:
        try:
            date(*map(int, re.findall(r"[0-9]+", found[1])))
            return found[1]
        except ValueError:
            return None
    return None


def _overview_lines(text: str):
    """Carry the most recent DART subheading; never borrow across headings."""
    heading = ""
    for line in text.splitlines():
        clean = _cell(line)
        match = re.match(r"^(?:[가-하]|[0-9]+|[IVX]+)\.\s*(.+)", clean)
        if match:
            heading = match[1]
        yield heading, clean


def _extract_founding_date(grouped: Sequence[Group]) -> tuple[str, Mapping[str, object]] | None:
    found = []
    for section, citation, text in grouped:
        if not section.endswith("회사의 개요"):
            continue
        for heading, line in _overview_lines(text):
            if re.search(r"자회사|종속|계열회사|현지법인", heading + " " + line):
                continue
            candidate = None
            if re.match(r"(?:설립일|회사성립연월일|창립일)", heading) and "당사는" in line:
                line = line[line.index("당사는"):]
            if line.startswith("|"):
                cells = [_cell(c) for c in line.strip("|").split("|")]
                if len(cells) == 2 and _compact(cells[0]) in {"설립일", "설립일자", "설립연월일", "회사성립연월일"}:
                    candidate = _valid_date_str(cells[1])
            elif line.startswith("당사는"):
                match = re.match(r"당사는(.{0,250}?)(?:설립되|설립하였|신설되)", line)
                if match and "상장" not in match[1] and "법인을" not in match[1]:
                    candidate = _valid_date_str(match[1])
            elif re.match(r"(?:설립일|회사성립연월일|창립일)", heading):
                # Bare dates are admissible only under the exact founding label,
                # not arbitrary nearby prose (e.g. a subsequent listing date).
                if re.fullmatch(r"[0-9]{4}(?:년\s*[0-9]{1,2}월\s*[0-9]{1,2}일|\.[0-9]{2}\.[0-9]{2})[.]?", line):
                    candidate = _valid_date_str(line)
            if candidate:
                report_year = re.search(r"\((20[0-9]{2})\.", str(citation["report_nm"]))
                if report_year and int(candidate[:4]) <= int(report_year[1]):
                    found.append((candidate, citation))
    return found[0] if len({_compact(value) for value, _ in found}) == 1 else None


def _extract_headquarters(grouped: Sequence[Group]) -> tuple[str, Mapping[str, object]] | None:
    overview_groups = [g for g in grouped if "1. 회사의 개요" in g[0] or g[0].endswith("회사의 개요")]
    search_groups = overview_groups if overview_groups else grouped

    for section, citation, text in search_groups:
        if not section.endswith("회사의 개요"):
            continue

        # 1. Table check: explicit row label matching 본점소재지 / 주소
        for heading, line in _overview_lines(text):
            if re.search(r"자회사|종속|계열|공장|지점|영업소", heading):
                continue
            if not line.startswith("|"):
                continue
            cells = [_compact(_cell(c)) for c in line.strip("|").split("|")]
            if len(cells) == 2 and cells[0] in {"주소", "본사의주소", "본점소재지", "본점주소"}:
                val = _cell(line.strip("|").split("|")[1])
                if _valid_address(val):
                    return val, citation

        # 2. Line check under parent address label
        for heading, line in _overview_lines(text):
            if re.search(r"자회사|종속|계열|공장|지점|영업소", heading):
                continue
            clean = _cell(line)
            matches = list(re.finditer(r"(?:본사의\s*주소|본점\s*소재지|본점\s*주소|(?<!홈페이지\s)(?<!이메일\s)(?<!인터넷\s)(?<!자회사\s)(?<!종속회사\s)주\s*소)\s*[:：]\s*([^\n]+)", clean))
            if not matches:
                continue
            for m in reversed(matches):
                prefix = clean[:m.start()]
                explicit_parent = m[0].startswith(("본사", "본점"))
                if ((_EXCLUDED_ADDR_PREFIX.search(prefix) and not explicit_parent)
                    or (explicit_parent and re.search(r"(?:자회사|종속회사|계열사)\s*$", prefix))):
                    continue
                val = m.group(1).strip()
                val = re.split(r"(?:\([0-9]+\)|[ㅇ•\-\s]+)*(?:전화번호|대표전화|홈페이지|전화|팩스)", val)[0].strip()
                val = re.sub(r"^[ㅇ•*\-\s]+", "", val).strip()
                if _valid_address(val):
                    return val, citation
    return None


def _valid_address(value: str) -> bool:
    return (len(value) >= 5 and _compact(value) not in _GENERIC_ABSENCE
            and not re.search(r"https?://|www\.|@|전화", value)
            and re.search(r"(?:특별시|광역시|특별자치|[가-힣]+[시도군구])", value) is not None)


def _extract_ceos(grouped: Sequence[Group], *, allow_unmarked_current: bool = False) -> tuple[list[tuple[str, Mapping[str, object], tuple[str, ...]]], bool]:
    roster_ceos: list[tuple[str, Mapping[str, object], bool]] = []
    text_indicates_joint = False

    for section, citation, text in grouped:
        if re.search(r"각자\s*대표|공동\s*대표|3인\s*대표", text):
            text_indicates_joint = True
        if "임원" not in section or "현황" not in section:
            continue

        lines = text.splitlines()
        for i, line in enumerate(lines):
            if not line.startswith("|"):
                continue
            header_cells = [_compact(_cell(c)) for c in line.strip("|").split("|")]
            if any(k in header_cells for k in ["사외이사후보자", "선임예정일", "선ㆍ해임예정일"]):
                continue
            if "성명" in header_cells and ("직위" in header_cells or "직책" in header_cells) and ("담당업무" in header_cells or "등기임원여부" in header_cells):
                name_idx = header_cells.index("성명")
                pos_idx = header_cells.index("직위") if "직위" in header_cells else header_cells.index("직책")
                duty_idx = header_cells.index("담당업무") if "담당업무" in header_cells else -1
                status_idx = header_cells.index("등기임원여부") if "등기임원여부" in header_cells else -1

                for dline in lines[i + 1 :]:
                    if not dline.startswith("|"):
                        break
                    dcells = [_cell(c) for c in dline.strip("|").split("|")]
                    if len(dcells) != len(header_cells):
                        continue
                    if dcells[0] in ("선임", "재선임", "해임") or "선임예정" in _compact(dline):
                        continue
                    name = dcells[name_idx]
                    if name in ("성명", "이름", "구분", "계", "합계") or re.match(r":?-+:?", name):
                        continue
                    name = re.sub(r"[\*\s]", "", name)
                    if not re.fullmatch(r"[가-힣]{2,5}", name):
                        continue

                    pos = dcells[pos_idx]
                    duty = dcells[duty_idx] if duty_idx >= 0 else ""
                    status = dcells[status_idx] if status_idx >= 0 else ""

                    combined = f"{pos} {duty}"
                    if _FORMER.search(combined):
                        continue
                    if status not in ("사내이사", "등기임원", "사내이사(상근)", "상근", ""):
                        continue
                    if _CEO_TERM.search(combined):
                        is_joint = bool(re.search(r"각자\s*대표|공동\s*대표", combined))
                        roster_ceos.append((name, citation, is_joint))
                break

    if not roster_ceos:
        return [], False

    seen: set[str] = set()
    deduped: list[tuple[str, Mapping[str, object], bool]] = []
    for name, c, is_j in roster_ceos:
        if name not in seen:
            seen.add(name)
            deduped.append((name, c, is_j))

    has_joint_flag = any(is_j for _, _, is_j in deduped) or text_indicates_joint
    if len(deduped) > 1 and not has_joint_flag and not allow_unmarked_current:
        return [], False

    res = []
    for name, c, is_j in deduped:
        lims = ("각자/공동 대표이사",) if (has_joint_flag and len(deduped) > 1) else ()
        if len(deduped) > 1 and not has_joint_flag:
            lims = ("임원 현황에 복수 대표이사가 기재되어 있으며 공동·각자 체제는 확정하지 않았습니다.",)
        res.append((name, c, lims))
    return res, has_joint_flag


def extract_company_profile(
    question: str,
    grouped: Sequence[Group],
    *,
    allow_unmarked_current_ceos: bool = False,
) -> tuple[ProfileFact, ...] | None:
    """Extract bounded company profile facts (founding date, headquarters, CEO).

    Returns a tuple of ProfileFact, or None if unsupported, ambiguous,
    conflicting, or ungrounded.
    """
    if not isinstance(question, str) or not grouped:
        return None
    if any(not _valid_group(g) for g in grouped):
        return None

    identity = lambda c: tuple(c.get(k) for k in ("corp_code", "corp_name", "rcept_no", "root_rcept_no", "latest_rcept_no", "report_nm", "correction_status"))
    if len({identity(c) for _, c, _ in grouped}) != 1:
        return None

    corp_name = str(grouped[0][1]["corp_name"])
    aliases = _company_aliases(corp_name, [t for _, _, t in grouped])
    if not _verify_question_company(question, aliases):
        return None
    years = set(re.findall(r"(?<![0-9])(20[0-9]{2})(?![0-9])", question))
    period = re.search(r"\((20[0-9]{2})\.[0-9]{2}\)", str(grouped[0][1]["report_nm"]))
    if years and (period is None or years != {period[1]}):
        return None

    is_founding = bool(re.search(r"설립(?:일|일자|연월일)?|창립(?:일|일자|연월일)?", question))
    is_hq = bool(re.search(r"본점|본사|소재지|주소", question))
    is_ceo = bool(re.search(r"대표\s*이사|대표|CEO", question))
    is_overview = bool(re.search(r"회사\s*개요|기업\s*개요|회사의\s*개요|어떤\s*회사|프로필", question))

    if not any([is_founding, is_hq, is_ceo, is_overview]):
        return None

    facts: list[ProfileFact] = []

    # 1. Founding date
    if is_founding or is_overview:
        founding = _extract_founding_date(grouped)
        if founding:
            val, cit = founding
            facts.append(ProfileFact("founding_date", val, MappingProxyType(dict(cit))))

    # 2. Headquarters
    if is_hq or is_overview:
        hq = _extract_headquarters(grouped)
        if hq:
            val, cit = hq
            facts.append(ProfileFact("headquarters", val, MappingProxyType(dict(cit))))

    # 3. CEO
    if is_ceo or is_overview:
        ceos, _ = _extract_ceos(grouped, allow_unmarked_current=allow_unmarked_current_ceos)
        for name, cit, lims in ceos:
            facts.append(ProfileFact("ceo", name, MappingProxyType(dict(cit)), limitations=lims))

    if not facts:
        return None

    return tuple(facts)
