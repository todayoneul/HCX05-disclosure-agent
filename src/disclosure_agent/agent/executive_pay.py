"""Conservative extraction of a single person's disclosed total compensation.

No I/O, model calls, or runner imports. The caller resolves company/period and
passes complete, source-ordered section triples from ONE verified filing. It
must reject read_section.truncated / remaining_parts before calling: a missing
whole row cannot be detected from plain text alone. Observable partial rows,
unclosed tables, mixed citations and conflicting duplicate facts fail closed.

Only explicit individual name/position/total headers are admitted. Never add
salary, bonus, average or group payments, or infer zero from non-disclosure.
For CEO questions the source role (or an exact-name executive roster row) must
prove the role at the filing's reference date, not today's current officeholder.
None means unsupported, ambiguous or insufficient evidence, not zero pay.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
import re
from types import MappingProxyType


Group = tuple[str, Mapping[str, object], str]
_NUMBER = re.compile(r"(?:0|[1-9][0-9]*|[1-9][0-9]{0,2}(?:,[0-9]{3})+)(?:\.[0-9]+)?")
_UNIT = re.compile(r"[（(]\s*단\s*위\s*[:：]\s*([^）)]+)[）)]")
_FORMER = re.compile(r"前|전임|퇴임|퇴직|사임|고문|자문|상담역|(?:^|\s)전\s*대표|former|retired", re.I)
_CEO = re.compile(r"대표\s*이사")


@dataclass(frozen=True)
class ExecutivePayFact:
    name: str
    role: str
    amount: str
    unit: str
    citation: Mapping[str, object]
    limitations: tuple[str, ...] = ()
    role_citation: Mapping[str, object] | None = None


@dataclass(frozen=True)
class _Table:
    rows: tuple[tuple[str, ...], ...]
    prefix: str
    closed: bool


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _cell(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<br\s*/?>", " ", value, flags=re.I)).strip()


def _tables(text: str) -> tuple[_Table, ...]:
    lines = text.splitlines()
    tables = []
    i = 0
    boundary = 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            i += 1
            continue
        start = i
        rows = []
        well_formed = True
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            line = lines[i].strip()
            well_formed &= line.endswith("|") and len(line) > 1
            cells = tuple(_cell(c) for c in line[1:-1].split("|"))
            if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                rows.append(cells)
            i += 1
        # Carry unit/title mini-tables forward, but never reuse a prior data
        # table's unit for a new table lacking its own unit declaration.
        prefix = "\n".join(lines[boundary:start])
        tables.append(_Table(tuple(rows), prefix, well_formed and i < len(lines)))
        if len(rows) > 1:
            boundary = i
    return tuple(tables)


def _valid_group(group: Group) -> bool:
    if not isinstance(group, (tuple, list)) or len(group) != 3:
        return False
    section, c, text = group
    keys = ("corp_code", "corp_name", "report_nm", "rcept_no", "root_rcept_no", "latest_rcept_no", "section")
    return (isinstance(text, str) and isinstance(section, str) and isinstance(c, Mapping)
            and all(isinstance(c.get(k), str) and c[k].strip() for k in keys)
            and c["section"] == section
            and all(re.fullmatch(r"[0-9]{14}", str(c[k])) for k in ("rcept_no", "root_rcept_no", "latest_rcept_no"))
            and c.get("is_latest") is True and c["latest_rcept_no"] == c["rcept_no"]
            and c.get("correction_status") in {"original", "linked"}
            and not c.get("truncated") and not c.get("remaining_parts"))


def _name_in_question(name: str, question: str) -> bool:
    return bool(re.search(r"(?<![가-힣A-Za-z])" + re.escape(name)
                          + r"(?=$|[^가-힣A-Za-z]|의\s|은\s|이\s)", question))


def _narrow_question(question: str, company: str, names: set[str]) -> bool:
    """Do not discard an unrecognised person/constraint from a CEO request."""
    remaining = question.replace(company, "")
    for name in names:
        remaining = remaining.replace(name, "")
    remaining = _compact(remaining)
    token = (r"대표이사|보수지급총액|보수총액|사업보고서|반기보고서|분기보고서|"
             r"개인별|개인|임원|이사|사장|기준|[0-9]{4}년?|[1-4]분기|"
             r"알려주세요|알려줘|얼마인가요|얼마인지|얼마야|얼마|무엇인가요|"
             r"인가요|입니다|은|는|이|가|의|을|를|요|[?.()]")
    return re.fullmatch(r"(?:" + token + r")+", remaining) is not None


def _header(row: tuple[str, ...]) -> tuple[int, int, int, int | None] | None:
    cells = [_compact(c) for c in row]
    names = [i for i, c in enumerate(cells) if c in {"이름", "성명"}]
    roles = [i for i, c in enumerate(cells) if c in {"직위", "직책"}]
    totals = [i for i, c in enumerate(cells) if c == "보수총액"]
    excluded = [i for i, c in enumerate(cells) if c == "보수총액에포함되지않는보수"]
    allowed = {"이름", "성명", "직위", "직책", "보수총액", "보수총액에포함되지않는보수"}
    if len(names) == len(roles) == len(totals) == 1 and len(excluded) <= 1 and all(c in allowed for c in cells):
        return names[0], roles[0], totals[0], excluded[0] if excluded else None
    return None


def _roster_roles(grouped: Sequence[Group]) -> dict[str, list[tuple[str, Mapping[str, object]]]]:
    result: dict[str, list[tuple[str, Mapping[str, object]]]] = {}
    for section, citation, text in grouped:
        if "임원" not in section or "현황" not in section or "보수" in section:
            continue
        for table in _tables(text):
            if not table.closed:
                continue
            for idx, row in enumerate(table.rows):
                cells = [_compact(c) for c in row]
                names = [i for i, c in enumerate(cells) if c in {"성명", "이름"}]
                roles = [i for i, c in enumerate(cells) if c in {"직위", "직책", "담당업무"}]
                if len(names) != 1 or not roles:
                    continue
                for data in table.rows[idx + 1:]:
                    if len(data) != len(row):
                        continue
                    name = data[names[0]]
                    role = " / ".join(data[i] for i in roles)
                    # A previous job in 주요경력 is not the present office.
                    status = " ".join(data[i] for i, c in enumerate(cells) if c in {"비고", "재직여부", "현직여부"})
                    if _FORMER.search(role + " " + status):
                        role = "퇴임 여부 확인 필요 / " + role
                    result.setdefault(name, []).append((role, citation))
                break
    return result


def _limits(text: str, prefix: str, name: str, excluded: str) -> tuple[str, ...]:
    limits = ["공시 기준일의 개인별 보수총액이며 미공개 대상의 보수를 0으로 해석할 수 없음"]
    if re.search(r"5\s*억\s*원?\s*이상", prefix):
        limits.append("5억원 이상 개인별 보수 공개 범위")
    if re.search(r"상위\s*5\s*명", prefix):
        limits.append("보수지급금액 상위 5명 공개 표 기준")
    if excluded not in {"", "-", "해당없음", "해당사항 없음"}:
        limits.append("보수총액에 포함되지 않는 보수: " + excluded)
    # Preserve explicit scope notes, not an inferred subtraction of retirement
    # income. The quoted total remains the filing's reported total.
    for line in text.splitlines():
        clean = _cell(line.strip("| "))
        for note in clean.split("※"):
            note = note.strip()
            if ("보수총액" in _compact(note) and "포함" in note
                    and (name in note or _compact(note).startswith("보수총액은"))
                    and len(note) <= 1000 and "|" not in note):
                limits.append(note)
    for table in _tables(text):
        if not table.closed or not table.rows:
            continue
        header = [_compact(c) for c in table.rows[0]]
        if header != ["이름", "보수의종류", "보수의종류", "총액", "산정기준및방법"]:
            continue
        units = {_compact(u) for u in _UNIT.findall(table.prefix)}
        if len(units) > 1 or not units <= {"원", "천원", "백만원", "억원"}:
            continue
        for row in table.rows[1:]:
            if (len(row) == 5 and _compact(row[0]).endswith(name)
                    and _compact(row[1]) == _compact(row[2]) == "퇴직소득"
                    and _NUMBER.fullmatch(row[3]) and Decimal(row[3].replace(",", "")) > 0):
                prefix_role = _compact(row[0])[:-len(name)]
                if prefix_role in {"", "대표이사", "前대표이사", "전대표이사", "고문", "자문", "자문역", "상담역", "사장", "사내이사", "이사", "부사장"}:
                    if units:
                        limits.append("동일 개인의 산정표에 퇴직소득 " + row[3] + next(iter(units)) + " 기재; 보수총액에서 임의 차감하지 않음")
                    else:
                        limits.append("동일 개인의 산정표에 퇴직소득 지급액이 기재되어 있음; 보수총액에서 임의 차감하지 않음")
    return tuple(dict.fromkeys(limits))


def extract_executive_pay(question: str, grouped: Sequence[Group]) -> tuple[ExecutivePayFact, ...] | None:
    """Return one safely bound disclosed person, or None; never a group total.

    ``role_citation`` is present when an additional roster section establishes
    CEO identity. The caller must cite/include that section in packed evidence.
    Preserve ``limitations`` and correction metadata when rendering the fact.
    """
    if not isinstance(question, str) or not grouped:
        return None
    q = _compact(question)
    if not re.search(r"보수(?:총액|지급총액)", q):
        return None
    if re.search(r"평균|급여|상여|퇴직|전\s*대표|前\s*대표|전임|합계|합산|전체|모두|각각|비교|차이|증감|및|하고|와\s|과\s|,|·", question):
        return None
    if any(not _valid_group(g) for g in grouped):
        return None
    identity = lambda c: tuple(c.get(k) for k in ("corp_code", "corp_name", "rcept_no", "root_rcept_no", "latest_rcept_no", "report_nm", "correction_status"))
    if len({identity(c) for _, c, _ in grouped}) != 1:
        return None
    facts = []
    for section, citation, text in grouped:
        if "보수" not in section:
            continue
        for table in _tables(text):
            headers = [(i, _header(row)) for i, row in enumerate(table.rows)]
            headers = [(i, h) for i, h in headers if h is not None]
            if not headers:
                continue
            idx, header = headers[0]
            assert header is not None
            units = _UNIT.findall(table.prefix)
            unit = _compact(units[-1]) if units else ""
            if (not table.closed or unit not in {"원", "천원", "백만원", "억원"}
                    or len({_compact(u) for u in units}) != 1):
                return None
            upper_allowed = {"이름", "성명", "직위", "직책", "보수", "보수총액", "보수총액에포함되지않는보수"}
            if any(len(row) != len(table.rows[idx]) or any(_compact(cell) not in upper_allowed for cell in row)
                   for row in table.rows[:idx]):
                return None
            ni, ri, ai, ei = header
            data_rows = table.rows[idx + 1:]
            if not data_rows:
                return None
            for row in data_rows:
                if _header(row) == header:
                    continue
                if len(row) != len(table.rows[idx]):
                    return None
                name, role, amount = row[ni], row[ri], row[ai]
                if not re.fullmatch(r"[가-힣]{2,5}", name) or not role or len(role) > 60:
                    return None
                # '-' and zero cannot prove non-disclosure is a zero payment.
                if not _NUMBER.fullmatch(amount) or Decimal(amount.replace(",", "")) <= 0:
                    return None
                facts.append(ExecutivePayFact(name, role, amount, unit,
                    MappingProxyType(dict(citation)), _limits(text, table.prefix, name, row[ei] if ei is not None else "")))
    if not facts:
        return None
    names = {f.name for f in facts if _name_in_question(f.name, question)}
    is_ceo = bool(_CEO.search(question))
    if len(names) > 1 or (not names and not is_ceo):
        return None
    if not _narrow_question(question, str(grouped[0][1]["corp_name"]), names):
        return None
    roster = _roster_roles(grouped)
    selected = []
    for fact in facts:
        if names and fact.name not in names:
            continue
        if is_ceo:
            roles = roster.get(fact.name, [])
            if _FORMER.search(fact.role) or any(_FORMER.search(r) for r, _ in roles):
                continue
            retired_note = any(fact.name in line and re.search(r"퇴임|퇴직하|사임|전\s*대표|前", line)
                               for _, _, text in grouped for line in text.splitlines()
                               if not line.strip().startswith("|") or line.strip().startswith("| ※"))
            if retired_note:
                continue
            if not _CEO.search(fact.role):
                proofs = [(r, c) for r, c in roles if _CEO.search(r)]
                if not proofs or any(not _CEO.search(r) for r, _ in roles):
                    continue
                role, c = proofs[0]
                fact = replace(fact, role=fact.role + " / " + role, role_citation=MappingProxyType(dict(c)))
        selected.append(fact)
    if len({f.name for f in selected}) != 1:
        return None
    # Validate all repetitions for that person, including rows whose role
    # disagreed with the selected current-CEO row.
    all_person = [f for f in facts if f.name == selected[0].name]
    if len({(f.role, f.amount, f.unit) for f in all_person}) != 1:
        return None
    limits = tuple(dict.fromkeys(s for f in all_person for s in f.limitations))
    return (replace(selected[0], limitations=limits),)
