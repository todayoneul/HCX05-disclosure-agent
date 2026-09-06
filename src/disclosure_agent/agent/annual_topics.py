"""Pure, bounded annual-topic extraction; no lookup, model or paraphrasing.

The caller owns issuer/year/receipt scope and correction disclosure. This helper
requires topic-appropriate business sections and complete original prose. Tables,
unresolved fragments and general market forecasts cannot fill a requested label.
FX is the supported risk family; other explicitly named risk families stay missing.
"""

from collections.abc import Iterable
import re

from disclosure_agent.context import EvidenceItem
from .answer_contract import citation_token


_REQUESTS = (
    ("연구개발 활동", r"연구\s*개발|R\s*&\s*D"),
    ("위험요인", r"위험|리스크|환율|외환"),
    ("향후 사업 계획", r"향후.*(?:사업|계획)|미래.*사업|사업\s*계획|신규\s*사업.*계획"),
)
_END = re.compile(r"(?:습니다|합니다|입니다|됩니다|하였다|했다|한다|된다|있다|없다|이다|않았다)\.")
_UNSAFE = re.compile(r"\[근거:|\[정정:|이전\s*지시|지시를?\s*무시|시스템\s*프롬프트|비밀키|API\s*키|secret|<[^>]+>", re.I)
_NAVIGATION = re.compile(r"☞|참고하시|참조하시|상세표|다음과\s*같|아래\s*표|상기\s*표|\.(?:jpg|jpeg|png|gif)\b", re.I)
_MID_FRAGMENT = re.compile(r"^(?:\d+[.)]|[가나다라마바사][.]|하며|으며|하고\s|위해\s|등을\s|수행하고|개발하고|이를\s|이러한\s|그러한\s|있는\s|없는\s|되는\s|된\s|위한\s|통해\s)")
_FX = re.compile(r"환율|외환|외화|환위험|선물환|통화\s*스왑")
_OTHER_RISK = re.compile(r"(?:신용|유동성|이자율|금리|가격|원자재)\s*(?:위험|리스크)")
_HEDGE = re.compile(r"선물환|통화\s*스[왑와]프?|환\s*헤지|환\s*헷지")
_MAX_SENTENCES = 3


def _sentences(text: str) -> list[str]:
    """Keep exact substrings, including line wraps; never splice across barriers."""
    blocks: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            blocks.append("\n".join(pending))
            pending.clear()

    for line in text.splitlines():
        stripped = line.strip()
        heading = not _END.search(line) and (
            re.match(r"^(?:#+|[가나다라마바사][.]|\d+[)])", stripped)
            or re.search(r"(?:개요|현황|조직|실적|활동|계획|위험|관리|사항|운영)$", stripped)
        )
        if not stripped or "|" in line or "\t" in line or _UNSAFE.search(line) or heading:
            flush()
        else:
            pending.append(line)
    flush()
    result = []
    for block in blocks:
        start = 0
        for end in _END.finditer(block):
            sentence = block[start:end.end()].strip()
            start = end.end()
            if (20 <= len(sentence) <= 2000 and not _MID_FRAGMENT.search(sentence)
                and not _NAVIGATION.search(sentence) and not _UNSAFE.search(sentence)):
                result.append(sentence)
    return result


def _usable(item: EvidenceItem) -> bool:
    c = item.citation
    return (
        isinstance(item.text, str)
        and c.get("is_latest") is True
        and c.get("correction_status") in {"original", "linked"}
        and c.get("latest_rcept_no") == c.get("rcept_no")
        and isinstance(c.get("rcept_no"), str)
        and re.fullmatch(r"[0-9]{14}", c["rcept_no"]) is not None
        and isinstance(c.get("corp_name"), str) and bool(c["corp_name"].strip())
        and isinstance(c.get("report_nm"), str)
        and re.search(r"사업보고서\s*\(\d{4}\.12\)", c["report_nm"]) is not None
        and isinstance(c.get("section"), str)
        and any(c["section"] == root or c["section"].startswith(root + " >")
                for root in ("II. 사업의 내용", "IV. 이사의 경영진단 및 분석의견"))
    )


def _research(sentence: str) -> bool:
    return (
        not re.search(r"연구개발비|비용|정부보조금|회계처리", sentence)
        and bool(re.search(
            r"(?:개발|연구|설계|시험|검증|상용화)(?:하|했|\s*중|"
            r"\s*(?:활동|과제|업무|프로젝트)?(?:을|를)?\s*(?:진행|수행|추진|완료))", sentence
        ))
    )


def _future(sentence: str, company: str) -> bool:
    issuer = re.match(
        r"^(?:또한,?\s*|아울러,?\s*)?(?:당사|회사|연결회사|" + re.escape(company)
        + r")(?=는|의|가|및|\s)", sentence
    )
    return bool(
        issuer
        and re.search(r"사업|제품|기술|설비|생산|서비스|공장|개발|진출|투자", sentence)
        and re.search(r"계획|예정|추진할|검토", sentence)
        and not re.search(r"것으로\s*(?:전망|예상)|매수|매도", sentence)
    )


def render_annual_topics(question: str, items: Iterable[EvidenceItem], *,
                         evidence_out: list[EvidenceItem] | None = None) -> tuple[list[str], list[str]]:
    """Return (cited original sentences, missing requested labels).

    At most three sentences are selected for research and plans, in source order.
    FX risk plus requested responses is all-or-none (one exposure and one explicit
    FX response, possibly the same sentence). No cross-chunk sentence stitching,
    implied hedge, inferred business plan, cost-table extraction or ownership
    substitution occurs. Original negation, numbers and future modality remain.
    A generic risk request can serve only a labeled FX subset and always reports
    "기타 위험요인" as missing; it cannot claim complete risk coverage.
    """
    requested = [label for label, pattern in _REQUESTS if re.search(pattern, question, re.I)]
    if not requested:
        return [], []
    candidates: dict[str, list[tuple[str, EvidenceItem]]] = {label: [] for label in requested}
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        if not isinstance(item, EvidenceItem) or not _usable(item):
            continue
        section = re.sub(r"\s+", "", item.citation["section"])
        for sentence in _sentences(item.text):
            for label in requested:
                appropriate = (
                    label == "연구개발 활동" and section.startswith("II.")
                    and "연구개발" in section and _research(sentence)
                    or label == "위험요인" and section.startswith("II.")
                    and bool(re.search(r"위험|리스크", section))
                    and bool(_FX.search(sentence)) and not _OTHER_RISK.search(sentence)
                    or label == "향후 사업 계획"
                    and bool(re.search(r"사업의개요|기타참고|향후|사업계획|신규사업|IV\.이사의경영진단및분석의견", section))
                    and _future(sentence, item.citation["corp_name"])
                )
                key = (label, sentence, citation_token(item.citation))
                if appropriate and key not in seen:
                    seen.add(key)
                    candidates[label].append((sentence, item))
    lines, missing = [], []
    for label in requested:
        selected = candidates[label][:_MAX_SENTENCES]
        if label == "위험요인":
            exposure = next((pair for pair in candidates[label]
                if re.search(r"노출되어|노출돼|손실.{0,20}발생|영향을?\s*(?:받|미)", pair[0])), None)
            responses = [pair for pair in candidates[label]
                         if re.search(r"관리|회피|대응|헤지|헷지|분산", pair[0])]
            response = next((pair for pair in responses if _HEDGE.search(pair[0])),
                            responses[0] if responses else None)
            needs_response = bool(re.search(r"대응|관리|방안|조치|회피|헤지", question))
            selected = []
            if exposure is not None and not _OTHER_RISK.search(question):
                if not needs_response or response is not None:
                    selected = [exposure]
                    if needs_response and response != exposure:
                        selected.append(response)
        if not selected:
            missing.append(label)
        rendered_label = label
        if label == "위험요인" and selected and not _FX.search(question):
            rendered_label = "환율위험(조회된 범위)"
            missing.append("기타 위험요인")
        for sentence, item in selected:
            lines.append(f"- {rendered_label}: {sentence} {citation_token(item.citation)}")
            if evidence_out is not None:
                # Preserve the exact selected substring and its original scope;
                # answer text is never reinterpreted as source evidence.
                evidence_out.append(EvidenceItem(
                    item.source_id + f"-annual-topic-{len(evidence_out)}", sentence,
                    item.citation, item.source_kind, item.priority + 100, item.rank))
    return lines, missing
