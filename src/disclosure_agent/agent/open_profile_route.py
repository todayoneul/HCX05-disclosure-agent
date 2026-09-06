"""Bounded annual-profile lookup, independent of the planner/model loop.

The caller validates every dispatch and release lineage. This layer additionally
pins issuer, annual period, receipt, exact section and source/evidence equality.
It returns only source-extractive facts; the normal answer validator still runs.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import re

from disclosure_agent.context import EvidenceItem
from disclosure_agent.tool_registry import ToolDispatchResult
from .answer_contract import citation_token
from .company_profile import extract_company_profile
from .narrative_quality import render_quality_narrative
from .annual_topics import render_annual_topics


@dataclass(frozen=True)
class OpenRequest:
    founding: bool = False
    headquarters: bool = False
    ceo: bool = False
    overview: bool = False
    business: bool = False
    additional_topics: tuple[str, ...] = ()


def open_request(question: str) -> OpenRequest | None:
    q = re.sub(r"\s+", "", question)
    if re.search(r"(?<![0-9])[0-9]{14}(?![0-9])|section|\b[IVX]+\.", question, re.I):
        return None
    # Leave multi-document, financial, event and explicitly constrained requests
    # to their existing specialized routes. A profile must not mask another ask.
    if re.search(r"비교|변화|변경|연혁|매출|이익|자산|부채|보수|배당|자본|자기주식|자사주|합병|투자|비율|수익|주가|시장|분기|반기|사업만|부문만|섹션|읽어", q):
        return None
    if len(set(re.findall(r"(?<![0-9])20[0-9]{2}(?![0-9])", q))) > 1:
        return None
    first = re.search(r"설립일|창립일|언제설립|본점|본사|소재지|대표이사|CEO|회사(?:의)?개요|기업개요|어떤회사|프로필|사업(?:의)?(?:내용|개요)|주요사업|핵심사업|어떤사업|연구개발|R&D|위험|리스크|향후사업계획", q, re.I)
    if first is None or not re.sub(r"[0-9년. ()]|사업보고서|기준", "", q[:first.start()]):
        return None
    request = OpenRequest(
        founding=bool(re.search(r"설립일|창립일|언제설립", q)),
        headquarters=bool(re.search(r"본점|본사|소재지", q)),
        ceo=bool(re.search(r"대표이사|CEO", q, re.I)),
        overview=bool(re.search(r"회사(?:의)?개요|기업개요|어떤회사|프로필", q)),
        business=bool(re.search(r"사업(?:의)?(?:내용|개요)|주요사업|핵심사업|어떤사업", q)),
        additional_topics=tuple(label for pattern, label in (
            (r"임직원수|직원수|종업원수|직원현황", "임직원 수"),
            (r"연구개발|R&D", "연구개발 활동"),
            (r"위험|리스크", "위험요인"),
            (r"향후.*(?:사업|계획)|사업계획", "향후 사업 계획"),
        ) if re.search(pattern, q, re.I)),
    )
    return request if any(vars(request).values()) else None


def supports_open_profile(schemas: list[dict]) -> bool:
    """Do not run exact-section orchestration with a search-only registry."""
    names = {schema.get("function", {}).get("name") for schema in schemas
             if isinstance(schema, dict) and isinstance(schema.get("function"), dict)}
    return {"resolve_company", "list_filings", "list_sections", "read_section"} <= names


@dataclass(frozen=True)
class OpenResult:
    answer: str = ""
    evidence: tuple[EvidenceItem, ...] = ()
    limitations: tuple[str, ...] = ()


def complete_roster_prefix(text: str, year: int) -> str | None:
    """Keep an explicitly closed year-end registered-director table, not a tail.

    A large employee/non-registered-director appendix may be truncated, but the
    issuer's first registered-director roster must have its own complete end.
    Never infer completeness from a final pipe at the end of the read budget.
    """
    for table in re.finditer(r"^\|[^\n]*(?:\n\|[^\n]*)*", text, re.MULTILINE):
        lines = table[0].splitlines(keepends=True)
        header_index = next((i for i, line in enumerate(lines) if all(label in re.sub(r"\s+", "", line)
                            for label in ("성명", "직위", "등기임원여부", "담당업무"))), None)
        if header_index is None:
            continue
        prefix = text[:table.start()] + "".join(lines[:header_index])
        if (re.search(r"임원\s*현황|등기임원", prefix) is None or "미등기임원" in prefix
            or re.search(rf"{year}\s*(?:년|[.])\s*12\s*(?:월|[.])\s*31", prefix) is None):
            return None
        if table.end() == len(text) or not text[table.end():].startswith("\n\n"):
            return None
        columns = len(lines[header_index].strip().split("|"))
        if any(not line.strip().endswith("|") or len(line.strip().split("|")) != columns
               for line in lines[header_index:]):
            return None
        return text[:table.end()]
    return None


def lookup_open_profile(
    question: str,
    request: OpenRequest,
    call: Callable[[str, dict], ToolDispatchResult | None],
    resolve: Callable[[ToolDispatchResult], dict[str, str] | None],
    name_source_company: Callable[[str, str], str],
) -> OpenResult:
    def unavailable(reason: str = "open_profile_evidence_unavailable") -> OpenResult:
        return OpenResult(limitations=(reason,))

    if re.search(r"20[0-9]{2}년(?:에)?\s*(?:공시|제출|접수)|오늘|실시간", question):
        return unavailable("open_profile_period_unsupported")
    resolved = call("resolve_company", {"query": question})
    company = resolve(resolved) if resolved is not None else None
    if company is None:
        return unavailable("open_profile_company_not_unique")
    corp = company["corp_code"]
    years = re.findall(r"(?<![0-9])(20[0-9]{2})(?![0-9])", question)
    year = int(years[0]) if years else None
    args = dict(corp_code=corp, base_month=12, doc_subtype="annual", latest_only=True, limit=50)
    if year is not None:
        args["base_year"] = year
    found = call("list_filings", args)
    rows = found.data if found else None
    if not isinstance(rows, (tuple, list)) or not rows:
        return unavailable()
    if any(not isinstance(row, Mapping) or row.get("corp_code") != corp
           or type(row.get("base_year")) is not int or row.get("base_month") != 12
           or row.get("doc_subtype") != "annual" for row in rows):
        return unavailable("open_profile_scope_mismatch")
    year = year if year is not None else max(row["base_year"] for row in rows)
    selected = [row for row in rows if row["base_year"] == year]
    if len(selected) != 1:
        return unavailable("open_profile_period_not_unique")
    receipt = selected[0].get("rcept_no")
    if not isinstance(receipt, str) or not re.fullmatch(r"[0-9]{14}", receipt):
        return unavailable()

    def scoped(item: EvidenceItem) -> bool:
        c = item.citation
        return (c["corp_code"] == corp and c["corp_name"] == company["corp_name"]
            and c["rcept_no"] == receipt and c["latest_rcept_no"] == receipt
            and c["is_latest"] is True and c["correction_status"] in {"original", "linked"}
            and "사업보고서" in c["report_nm"]
            and re.search(rf"\({year}\s*\.\s*12\)", c["report_nm"]) is not None)

    sections = call("list_sections", dict(rcept_no=receipt, limit=50))
    if sections is None or not isinstance(sections.data, (tuple, list)):
        return unavailable()
    paths = [row.get("path") for row in sections.data if isinstance(row, Mapping)]
    paths = [path for path in paths if isinstance(path, str)]
    wanted = []
    if request.founding or request.headquarters or request.overview:
        wanted += [p for p in paths if p.endswith("1. 회사의 개요") or p == "I. 회사의 개요"][:1]
    if "연구개발 활동" in request.additional_topics:
        wanted += [p for p in paths if p.startswith("II.") and "연구개발" in p][:1]
    if "위험요인" in request.additional_topics:
        wanted += [p for p in paths if p.startswith("II.") and "위험" in p][:1]
    if "향후 사업 계획" in request.additional_topics:
        wanted += [p for p in paths if p.startswith("IV.") and "경영진단" in p][:1]
        wanted += [p for p in paths if p.startswith("II.") and "기타 참고" in p][:1]
    if request.business or request.overview:
        # Prefer the root overview before per-segment/price tables. Some issuers
        # put the true overview directly in II, others in II > 1.
        business_paths = [p for p in paths if p == "II. 사업의 내용" or
            (p.startswith("II. 사업의 내용 > 1.") and "사업의 개요" in p)]
        primary = [p for p in business_paths if "(금융업)" not in p]
        wanted += sorted(primary or business_paths, key=lambda p: (p != "II. 사업의 내용", len(p)))[:2]
        wanted += [p for p in paths if p.startswith("II. 사업의 내용 > 2.")
                   and "주요 제품" in p and "(금융업)" not in p][:1]
        if primary:
            wanted += [p for p in business_paths if "(금융업)" in p][:1]
    if request.ceo:
        roster = [p for p in paths if "임원" in p and "현황" in p and "보수" not in p]
        if not roster:
            search = call("search_chunks", dict(query="대표이사 성명 직위 담당업무", corp_code=corp,
                base_year=year, base_month=12, doc_subtype="annual", latest_only=True,
                path_hint="임원 및 직원", k=3))
            if search and search.evidence and all(scoped(e) for e in search.evidence):
                roster = list(dict.fromkeys(e.citation["section"] for e in search.evidence
                    if "임원" in e.citation["section"] and "현황" in e.citation["section"]
                    and "보수" not in e.citation["section"]))
        if len(roster) == 1:
            wanted += roster
    groups = []
    evidence = []
    for path in dict.fromkeys(wanted):
        full = call("read_section", dict(rcept_no=receipt, path=path, max_chars=12000))
        if full is None or not isinstance(full.data, Mapping) or not full.evidence:
            continue
        data = full.data
        if (data.get("path") != path or not isinstance(data.get("text"), str)
            or data["text"] != "\n".join(e.text for e in full.evidence)
            or not all(scoped(e) and e.citation["section"] == path
                       and e.citation == full.evidence[0].citation for e in full.evidence)):
            return unavailable("open_profile_scope_mismatch")
        # Profile tables must be complete. For business prose, complete source
        # sentences can safely be extracted from a bounded section prefix.
        section_text = data["text"]
        source_items = full.evidence
        incomplete = (data.get("truncated") is not False
            or type(data.get("remaining_parts")) is not int or data["remaining_parts"] != 0)
        if incomplete and "임원" in path and complete_roster_prefix(section_text, year) is None:
            # One bounded continuation, rereading any partially consumed chunk.
            # The second page must close the section; never splice a gap or
            # merge another filing into a superficially plausible roster.
            parts = data.get("chunks")
            next_part = data.get("next_part")
            if (isinstance(parts, (tuple, list)) and len(parts) == len(source_items)
                and all(isinstance(p, Mapping) and type(p.get("part")) is int
                        and p.get("chunk_id") == e.source_id and p.get("text") == e.text
                        for p, e in zip(parts, source_items))
                and [p["part"] for p in parts] == list(range(1, len(parts) + 1))
                and type(next_part) is int and next_part in {len(parts), len(parts) + 1}):
                page = call("read_section", dict(rcept_no=receipt, path=path, max_chars=12000, part_from=next_part))
                if page and isinstance(page.data, Mapping) and page.evidence:
                    pd = page.data
                    page_parts = pd.get("chunks")
                    if (pd.get("path") != path or pd.get("text") != "\n".join(e.text for e in page.evidence)
                        or not all(scoped(e) and e.citation == full.evidence[0].citation for e in page.evidence)
                        or not isinstance(page_parts, (tuple, list)) or len(page_parts) != len(page.evidence)
                        or not all(isinstance(p, Mapping) and p.get("chunk_id") == e.source_id
                            and p.get("text") == e.text for p, e in zip(page_parts, page.evidence))
                        or [p.get("part") for p in page_parts] != list(range(next_part, next_part + len(page_parts)))
                        or (next_part == len(parts) and not page.evidence[0].text.startswith(source_items[-1].text))):
                        return unavailable("open_profile_scope_mismatch")
                    if pd.get("truncated") is False and type(pd.get("remaining_parts")) is int and pd["remaining_parts"] == 0:
                        source_items = (*source_items[:next_part - 1], *page.evidence)
                        section_text = "\n".join(e.text for e in source_items)
                        incomplete = False
        if "사업의 내용" not in path and not path.startswith("IV.") and incomplete:
            prefix = complete_roster_prefix(section_text, year) if "임원" in path else None
            if prefix is None:
                continue
            section_text = prefix
            item = full.evidence[0]
            evidence.append(EvidenceItem(item.source_id + "-complete-roster", prefix, item.citation,
                                         item.source_kind, item.priority, item.rank))
        else:
            evidence.extend(source_items)
        groups.append((path, full.evidence[0].citation, section_text))
    if not groups:
        return unavailable()
    facts = extract_company_profile(question, groups, allow_unmarked_current_ceos=True) or ()
    labels = {"founding_date": "설립일", "headquarters": "본점 소재지", "ceo": "대표이사"}
    lines = [f"- {labels[f.kind]}: {f.value}. {citation_token(f.citation)}" for f in facts]
    needed = {kind for kind, requested in (("founding_date", request.founding or request.overview),
        ("headquarters", request.headquarters or request.overview), ("ceo", request.ceo)) if requested}
    missing = [labels[k] for k in sorted(needed - {f.kind for f in facts})]
    if request.business or request.overview:
        narrative = render_quality_narrative(question, groups,
            ((year, 12, "annual", f"{year}년 사업보고서"),),
            name_source_company=name_source_company, allow_unconstrained=True)
        if narrative and "검색된 사업 본문에서" not in narrative:
            lines.append(narrative)
        else:
            missing.append("주요 사업")
    topic_evidence: list[EvidenceItem] = []
    topic_lines, topic_missing = render_annual_topics(question, evidence, evidence_out=topic_evidence)
    evidence[:0] = topic_evidence
    lines.extend(topic_lines)
    missing.extend(topic_missing)
    if "임직원 수" in request.additional_topics:
        missing.append("임직원 수")
    if not lines:
        return unavailable()
    if missing:
        lines.append("확인하지 못한 항목: " + ", ".join(missing) + ".")
    lead = f"{company['corp_name']} — 사업보고서 ({year}.12) 기재 기준"
    if not years:
        lead += " (제공된 연간 공시 중 최신 자료이며 현재 정보의 확인은 아닙니다)"
    return OpenResult(lead + "\n" + "\n".join(lines), tuple(evidence),
                      ("open_profile_partial",) if missing else ())
