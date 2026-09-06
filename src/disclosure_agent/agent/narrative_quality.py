"""Bounded, extractive business summaries and aligned periodic narratives.

This module performs no retrieval, calculation, model call or file access.
``grouped`` contains independently grounded (section, citation, text) triples;
``documents`` contains (year, month, subtype, display_label) tuples in requested
order. Labels are reconstructed, never copied into the answer from user text.

Return None for an unhandled request or inadmissible/ambiguous input. A string,
including a cited evidence/format limitation, is terminal: callers must NOT
replace it with the legacy first-paragraph excerpt or a model completion.
The caller remains responsible for resolved-company binding, verified release
lineage, correction disclosures and final claim/citation validation. Apply
presentation limits to the final response too; do not prepend a generic lead-in.

Only complete source sentences are extracted. Whitespace, DART headings and
navigation are cleaned, but numbers, names, negation and modality are not
rewritten. ``name_source_company(sentence, corp_name)`` may be supplied by the
runner to reuse its existing conservative issuer-pronoun normalizer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
import re

from .answer_contract import citation_token


Document = tuple[int, int, str, str]
Group = tuple[str, Mapping[str, object], str]
Topic = tuple[str, str]

# These are topic-recognition aliases, never substitutions in source text.
# An unrecognised topic cannot establish cross-document equivalence.
_DOMAINS = {
    "메모리": r"DRAM|NAND|HBM|DDR[0-9]*|메모리",
    "완제품": r"(?<![A-Za-z])TV(?![A-Za-z])|스마트폰|냉장고|세탁기|에어컨|전자제품",
    "디스플레이": r"OLED|디스플레이",
    "파운드리": r"Foundry|파운드리",
    "CIS": r"(?<![A-Za-z])CIS(?![A-Za-z])|CMOS Image Sensor",
    "항공": r"항공",
    "방산": r"방산|자주포|군수장비",
    "시큐리티": r"시큐리티|CCTV",
    "IT 서비스": r"IT\s*서비스|정보시스템",
    "산업용장비": r"산업용\s*(?:장비|기계)|산업기계",
    "해양": r"조선|해양제품|해양사업|선박",
    "우주": r"(?<!항공)우주사업|위성시스템",
    "DX": r"(?<![A-Za-z])DX(?![A-Za-z])",
    "DS": r"(?<![A-Za-z])DS(?![A-Za-z])",
    "SDC": r"(?<![A-Za-z])SDC(?![A-Za-z])",
    "Harman": r"Harman|하만|카오디오|디지털\s*콕핏",
    "자동차": r"자동차|(?<!철도)(?<!철도 )차량",
    "배터리": r"배터리|이차전지|2차전지",
    "의약품": r"의약품|제약",
    "게임": r"게임",
    "음악": r"음악|음반|공연|뮤직",
    "콘텐츠": r"콘텐츠|컨텐츠|웹툰|스토리IP",
    "물류": r"물류|해운|컨테이너|벌크|화물\s*운송|해상운송|물류서비스",
    "원자력": r"원자력|원전|원자로|증기발생기|발전설비|복합화력|스팀터빈|발전플랜트|해상풍력",
    "인터넷/플랫폼": r"포털|검색|커머스|핀테크|메신저|플랫폼|소프트웨어|모바일|카카오톡",
    "철강": r"철강|제철|강판|열연|냉연|후판",
    "지주": r"지주회사|포트폴리오",
    "금융": r"금융|할부금융|신용카드|리스(?:금융|업|사업|상품|거래)|(?<![가-힣])리스(?![가-힣])|증권|보험|은행|여신전문",
}
_PATTERNS = {key: re.compile(value, re.I) for key, value in _DOMAINS.items()}
_FAMILIES = {"business": "주요 제품·사업", "composition": "사업부문 매출 구성",
             "market": "시장·수요", "sales": "판매망", "investment": "투자 계획"}
_END = re.compile(r"[다요음임함됨]\.[\"”’']?")
_NAVIGATION = re.compile(r"☞|참고하시|참조하시|참고\s*바랍니다|상세한?\s*(?:내용|사항).*참고|자세한.*참고")
_INSTRUCTION = re.compile(r"\[근거:|\[정정:|이전\s*(?:지시|규칙)|시스템\s*프롬프트|비밀키|API\s*키|secret|Authorization", re.I)
_ACTION = re.compile(
    r"생산|판매|개발|양산|영위|주력\s*제품|핵심\s*사업|사업.*병행|"
    r"제작|공급|운영|전개|수행|물류서비스|서비스를\s*제공|솔루션을\s*제공|"
    r"(?:사업)?부문으로\s*(?:구성|구분)|부문은.*구성|사업으로\s*구성|콘텐츠로\s*구성|사업\s*포트폴리오|"
    r"매출을\s*창출|주요\s*사업|핵심\s*자산|역할을\s*수행|대표적인\s*금융기업"
)
_TABLE_LEAD_IN = re.compile(
    r"(?:재무정보|매출액\s*(?:및\s*그\s*비중)?|실적|현황|세부\s*내용|내역|주요\s*사항|영업수익\s*대비\s*비중)은?\s*(?:다음과|아래와)\s*같|"
    r"(?:다음과|아래와)\s*같(?:습니다|으며|고)|"
    r"세부\s*내용은\s*(?:다음|아래)|"
    r"(?:다음|아래)의\s*표|"
    r"기재를?\s*생략|"
    r"(?:다음과\s*같이|아래와\s*같이)\s*(?:요약|정리|구분)"
)
_PERFORMANCE_NUMBERS = re.compile(
    r"(?:판매\s*실적|누계\s*약|누적\s*매출액|연결\s*매출과|영업이익을\s*기록|매출을\s*기록|수주잔고는|매출\s*실적을\s*바탕으로).*(?:기록|확보|달성)|"
    r"(?:매출액은|매출은)\s*[0-9,]+\s*(?:조|억|만)\s*원"
)
_ASPIRATION_STRATEGY = re.compile(
    r"성장을\s*목표로.*(?:투자|강화)|"
    r"끊임없는\s*도전|도전을\s*멈추지|도전을\s*이어가고|도약의\s*기회|"
    r"중장기\s*전략|'2030\s*전략'|2030\s*전략|전략을\s*통해|"
    r"경영\s*전략의\s*핵심|지속가능성.*견인|"
    r"기업이\s*되도록\s*노력|노력을\s*기울이고|노력해\s*나갈\s*것입니다|노력하겠습니다|"
    r"경쟁력을\s*확보하여.*공급하고|역할을\s*수행하였습니다|"
    r"사람을\s*이해하는\s*기술|존재이유|핵심가치|미션\s*아래|비전으로|슬로건|"
    r"풍요로운\s*미래|새로운\s*내일|더\s*나은\s*(?:세상|미래|삶)|따뜻한\s*기술|인류의\s*삶|"
    r"브랜드\s*(?:슬로건|아이덴티티|캠페인)|고객과\s*(?:함께|사회와\s*함께)"
)
_CORE_BUSINESS_STRUCTURE = re.compile(
    r"(?:사업은|사업부문은|사업을|사업으로|부문으로|포트폴리오를|기반으로|통해)\s*"
    r"(?:운영|영위|구성|구분|전개|수행|창출|제공|제안|설계|제작|생산|판매)|"
    r"(?:제조[ㆍ·,/\s]*(?:및\s*)?판매|생산[ㆍ·,/\s]*(?:및\s*)?판매|설계[·ㆍ]제작|제작[·ㆍ]공급|연구,\s*개발,\s*제조,\s*판매|개발자\s*역할)|"
    r"(?:물류서비스|솔루션을)\s*(?:제공|제안)|"
    r"사업\s*포트폴리오|주요\s*사업|핵심\s*사업|매출을\s*창출"
    r"|(?:사업)?부문으로\s*(?:구성|구분)|부문은.*구성|콘텐츠로\s*구성"
    r"|제작(?:하여|/|·).*공급|설계.*제작"
)
_OVERARCHING_ENTITY = re.compile(
    r"연결실체|당사와\s*연결종속|당사\s*및\s*(?:당사의\s*)?종속|연결기준\s*사업부문|"
    r"총\s*[0-9]+개의?\s*사업부문|사업부문은\s*매출의\s*성격에\s*따라|다각화된\s*사업\s*포트폴리오|"
    r"Set\s*사업|부품\s*사업|"
    r"대표적인\s*금융기업|국내\s*1위|대표\s*메신저|인터넷\s*검색\s*포털|지주회사로\s*전환|"
    r"종합해운물류기업|대표적인\s*발전설비|주조/단조를\s*기반|배터리\s*기술\s*개발을\s*핵심\s*전략"
)
_ORGANIZATION = re.compile(
    r"[0-9,]+\s*개(?:의)?\s*(?:종속기업|연구개발법인|생산기지)|"
    r"종속기업으로\s*구성|지역별로\s*보면|해외\s*[（(]|"
    r"(?:본사|국내\s*종속기업).*(?:사업장|종속기업).*구성|"
    r"[0-9,]+\s*개(?:의)?[^.]{0,40}(?:법인|종속기업)|"
    r"^(?:미주|유럽|아시아|중동|아프리카).*법인"
)
_COUNT_VALUES = {"한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3,
                 "네": 4, "넷": 4, "다섯": 5, "여섯": 6, "일곱": 7,
                 "여덟": 8, "아홉": 9, "열": 10}
_NUMBER = r"(?:[0-9]+|여덟|일곱|아홉|다섯|여섯|하나|둘|셋|넷|한|두|세|네|열)"
_COUNT = re.compile(
    rf"(?P<amount>두\s*세|한\s*두|서너|{_NUMBER}(?:\s*[~∼～\-–]\s*{_NUMBER})?)"
    r"\s*(?P<unit>문장|항목|가지|문단)(?P<tail>\s*(?:이내|이하|까지|정도))?"
)


@dataclass(frozen=True)
class _Limit:
    unit: str
    minimum: int
    maximum: int


@dataclass(frozen=True)
class _Sentence:
    text: str
    citation: Mapping[str, object]
    topics: frozenset[Topic]
    domains: frozenset[str]
    order: tuple[int, str, int, str]
    qualifiers: tuple[str, ...] = ()
    caveats: tuple[tuple[str, Mapping[str, object]], ...] = ()


def _limits(question: str) -> tuple[_Limit, ...]:
    found = []
    for match in _COUNT.finditer(question):
        amount = re.sub(r"\s+", "", match["amount"])
        if amount in {"두세", "한두", "서너"}:
            lower, upper = {"두세": (2, 3), "한두": (1, 2), "서너": (3, 4)}[amount]
        else:
            parts = re.split(r"[~∼～\-–]", amount)
            values = [int(part) if part.isdigit() else _COUNT_VALUES[part] for part in parts]
            lower, upper = values[0], values[-1]
        if match["tail"]:
            lower = 1
        found.append(_Limit(match["unit"], lower, upper))
    return tuple(found)


def _domains(text: str) -> frozenset[str]:
    found = {key for key, pattern in _PATTERNS.items() if pattern.search(text)}
    if re.search(r"유럽\s*[ㆍ·,/\-]\s*CIS|독립국가연합", text, re.I):
        found.discard("CIS")
    return frozenset(found)


def _document_label(doc: Document) -> str:
    year, month, subtype, _ = doc
    kind = "사업보고서" if subtype == "annual" else "반기보고서" if subtype == "half" else f"{month // 3}분기보고서"
    return f"{year}년 {kind}"


def _matches(citation: Mapping[str, object], doc: Document) -> bool:
    year, month, subtype, _ = doc
    report = str(citation["report_nm"])
    kind = {"annual": "사업보고서", "half": "반기보고서", "quarter": "분기보고서"}[subtype]
    return kind in report and re.search(rf"\({year}\s*\.\s*{month:02d}\)", report) is not None


def _valid_citation(section: str, citation: Mapping[str, object]) -> bool:
    required = ("corp_code", "corp_name", "report_nm", "rcept_no", "root_rcept_no", "latest_rcept_no", "section")
    return (
        all(isinstance(citation.get(key), str) and str(citation[key]).strip() for key in required)
        and citation.get("section") == section
        and re.fullmatch(r"[0-9]{14}", str(citation["rcept_no"])) is not None
        and re.fullmatch(r"[0-9]{14}", str(citation["root_rcept_no"])) is not None
        and citation.get("is_latest") is True
        and citation.get("latest_rcept_no") == citation.get("rcept_no")
        and citation.get("correction_status") in {"original", "linked"}
    )


def _source_sentences(text: str) -> tuple[str, ...]:
    # Unwrap a genuinely single-cell prose block before expanding its BRs.
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("|"):
            cells = line.strip("|").split("|")
            if len(cells) != 1 or _END.search(cells[0]) is None:
                lines.append("\x00")
                continue
            line = cells[0].strip()
        lines.extend(re.split(r"<br\s*/?>", line, flags=re.I))
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line or re.search(r"\.(?:jpg|jpeg|png|gif)$", line, re.I):
            cleaned.append("\x00")
            continue
        if line.startswith(("※", "*")):
            cleaned.append("\x00")
            cleaned.append(line)
            cleaned.append("\x00")
            continue
        line = re.sub(r"^[가나다라마바사]\.\s*", "", line)
        line = re.sub(r"^주요\s*제품\s*매출(?=당사)", "", line)
        if re.fullmatch(r"(?:주요 제품.*|사업의 개요|사업의 내용|\(?영업의 개황 등\)?|\[.*\])", line) and not _END.search(line):
            continue
        cleaned.append(line)
    prose = re.sub(r"\s+", " ", " ".join(cleaned)).strip()
    result = []
    for block in prose.split("\x00"):
        start = 0
        for match in _END.finditer(block):
            sentence = block[start:match.end()].strip()
            start = match.end()
            if not sentence or _NAVIGATION.search(sentence) or _INSTRUCTION.search(sentence):
                continue
            if "|" in sentence or len(sentence) > 2000:
                continue
            result.append(sentence)
    # An unterminated tail is never promoted to a completed source claim.
    return tuple(result)


def _topics(text: str, domains: frozenset[str]) -> frozenset[Topic]:
    if not domains or _ORGANIZATION.search(text) or re.search(r"생산공정.*원재료|원재료.*(?:웨이퍼|Wafer|PCB)", text, re.I):
        return frozenset()
    if re.search(r"판매\s*가격|판매\s*단가|가격\s*변동", text):
        return frozenset()
    if not re.search(r"당사|계획입니다|계획하고", text) and re.search(r"경쟁력.*(?:될|것)|중요.*(?:것입니다|것으로)|향후.*(?:전망|예상)", text):
        return frozenset()
    if re.search(r"판매법인|판매망|판매경로|영업망|유통망", text):
        family = "sales"
    elif not re.search(r"부문(?:은|으로)|사업(?:은|으로)|영위|전개", text) and re.search(r"시장|업황|업계|수요", text) and not re.search(r"당사|당사의|당사는", text):
        family = "market"
    elif re.search(r"(?:투자|증설).*(?:계획|예정)|설비투자", text):
        family = "investment"
    elif _ACTION.search(text):
        family = "business"
    elif re.search(r"매출|비중", text) and re.search(r"[0-9]", text):
        family = "composition"
    else:
        return frozenset()
    anchors = set(domains)
    if "메모리" in anchors:
        products = [value for value in ("DRAM", "NAND", "HBM", "DDR") if re.search(value, text, re.I)]
        if products:
            anchors.remove("메모리")
            anchors.update("메모리/" + value for value in products)
    return frozenset((family, domain) for domain in anchors)


def _qualifiers(text: str) -> tuple[str, ...]:
    """Keep material accounting qualifications when extracting prose figures."""
    found = []
    for line in text.splitlines():
        for cell in line.split("|"):
            if re.search(r"내부거래|연결\s*기준|별도\s*기준", cell):
                cleaned = re.sub(r"^[\s※*]+", "", cell).strip()
                if _END.search(cleaned) and not _INSTRUCTION.search(cleaned):
                    found.append(cleaned.rstrip("."))
    return tuple(dict.fromkeys(found))


def _conflicting_percent(sentence: str, text: str) -> bool:
    """Reject a clearly conflicting named segment percentage in prose/table.

    This is a narrow conflict guard, not a general financial table parser.
    It does not convert units or derive a percentage from table amounts.
    """
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        label = re.sub(r"\s*부문$", "", cells[0]).strip()
        table_percent = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)%", cells[-1])
        if not table_percent or not label or len(label) > 40:
            continue
        prose_percent = re.search(re.escape(label) + r"[^%]*?\(([0-9]+(?:\.[0-9]+)?)%\)", sentence)
        if prose_percent and Decimal(table_percent[1]) != Decimal(prose_percent[1]):
            return True
    return False


def _issuer_name(text: str, company: str) -> str:
    # Only an unambiguous leading issuer pronoun is rewritten by default.
    # The caller can supply its more complete, tested Korean normalizer.
    tail = ord(company[-1]) - 0xAC00 if company else 0
    consonant = 0 <= tail < 11172 and tail % 28 != 0
    topic_consonant = "은" if consonant else "는"
    subj_consonant = "이" if consonant else "가"
    with_consonant = "과" if consonant else "와"
    for source, target in (
        ("당사는", company + topic_consonant),
        ("당사의", company + "의"),
        ("당사가", company + subj_consonant),
        ("당사와", company + with_consonant),
        ("당사 및", company + " 및"),
        ("당사는 물론", company + topic_consonant + " 물론"),
    ):
        if text.startswith(source):
            return target + text[len(source):]
    return text


def _limitation(message: str, citations: Sequence[Mapping[str, object]]) -> str:
    tokens = list(dict.fromkeys(citation_token(citation) for citation in citations))
    return message + (" " + " ".join(tokens) if tokens else "")


def _is_business_overview_query(question: str) -> bool:
    cleaned = re.sub(r"(?:사업|분기|반기)\s*보고서", "", question)
    return bool(re.search(
        r"사업\s*(?:의\s*)?(?:내용|개요|부문|영역|분야|활동|현황|포트폴리오|변화|구성|흐름)?|"
        r"주요\s*사업|핵심\s*사업|영위|어떤\s*(?:회사|기업|사업)|무슨\s*(?:회사|기업|사업)|"
        r"회사\s*(?:의\s*)?개요|주요\s*제품|서비스",
        cleaned
    ))


def render_quality_narrative(
    question: str,
    grouped: Sequence[Group],
    documents: Sequence[Document],
    *,
    name_source_company: Callable[[str, str], str] | None = None,
    allow_unconstrained: bool = False,
) -> str | None:
    """Extract constrained summaries or same-topic multi-document comparisons.

    None means not handled/unsafe; a cited limitation string is an intentional
    answer. Documents must already have been resolved by the runner (including
    shorthand periods). Explicit output counts apply to extracted sentences or
    topic rows, never to the number of citations. No facts are padded to meet an
    exact count: insufficient evidence is disclosed within the upper bound.
    """
    if not isinstance(question, str) or len(question) > 10_000 or len(grouped) > 100 or len(documents) > 8:
        return None
    docs: list[Document] = []
    for doc in documents:
        if (not isinstance(doc, (tuple, list)) or len(doc) != 4
            or type(doc[0]) is not int or not 1900 <= doc[0] <= 9999
            or type(doc[1]) is not int
            or doc[2] not in {"annual", "half", "quarter"}
            or doc[1] not in {"annual": {12}, "half": {6}, "quarter": {3, 9, 12}}[doc[2]]):
            return None
        normalized = (doc[0], doc[1], doc[2], "")
        if normalized not in docs:
            docs.append(normalized)
    limits = _limits(question)
    if any(not 1 <= limit.minimum <= limit.maximum <= 12 for limit in limits):
        return None
    rows: list[Group] = []
    for row in grouped:
        if not isinstance(row, (tuple, list)) or len(row) != 3:
            return None
        section, citation, text = row
        if not isinstance(section, str) or not isinstance(citation, Mapping) or not isinstance(text, str) or len(text) > 100_000:
            return None
        if "사업의 내용" not in section and "사업의 개요" not in section:
            continue
        if not _valid_citation(section, citation):
            return None
        if not docs or any(_matches(citation, doc) for doc in docs):
            rows.append((section, citation, text))
    if not rows or len({(c["corp_code"], c["corp_name"]) for _, c, _ in rows}) != 1:
        return None
    company = str(rows[0][1]["corp_name"])
    scope_question = question.replace(company, "")
    scopes = _domains(scope_question)
    scope_exclusive = bool(re.search(r"만(?:\s|로|을|은|$)", scope_question))
    unknown_scope = bool(re.search(r"사업만|부문만|분야만", re.sub(r"\s+", "", scope_question))) and not scopes
    multi = len(docs) > 1
    is_overview = allow_unconstrained and not multi and _is_business_overview_query(question)
    is_product_change = bool(re.search(r"(?:핵심|주요)\s*(?:제품|사업)", question) and re.search(r"변화|변동|비교|어떻게", question))
    if not multi and not limits and not scopes and not unknown_scope and not is_overview:
        return None
    naming = name_source_company or _issuer_name
    buckets: list[list[Group]] = [[row for row in rows if _matches(row[1], doc)] for doc in docs] if docs else [rows]
    if any(len({str(c["rcept_no"]) for _, c, _ in bucket}) > 1 for bucket in buckets):
        return None
    citations = [sorted(bucket, key=lambda row: (row[0], row[2]))[0][1] for bucket in buckets if bucket]
    if any(not bucket for bucket in buckets):
        missing = ", ".join(_document_label(docs[i]) for i, bucket in enumerate(buckets) if not bucket)
        return _limitation(f"검색된 사업 본문에서 {missing}의 근거가 부족하여 요청한 문서 간 사업 변화를 비교할 수 없습니다.", citations)

    candidates: list[list[_Sentence]] = []
    for bucket in buckets:
        entries = []
        seen = set()
        # Keep a same-report discontinuation note with the affected business.
        # Its other business names are qualification context, not extra topics.
        notices = []
        notice_keys = set()
        for _, citation, text in sorted(bucket, key=lambda row: (row[0], row[2])):
            for statement in _source_sentences(text):
                key = (statement, citation_token(citation))
                if "중단사업" in statement and _domains(statement) and key not in notice_keys:
                    notice_keys.add(key)
                    notices.append((statement.lstrip("※* "), citation))
        for section, citation, text in sorted(bucket, key=lambda row: (row[0], row[2])):
            for index, sentence in enumerate(_source_sentences(text)):
                domains = _domains(sentence)
                topics = _topics(sentence, domains)
                if not topics and not domains and is_overview and _ACTION.search(sentence):
                    if not _ORGANIZATION.search(sentence) and not re.search(r"생산공정.*원재료|원재료.*(?:웨이퍼|Wafer|PCB)|판매\s*가격|가격\s*변동", sentence):
                        topics = frozenset({("business", "사업")})
                        if not domains:
                            domains = frozenset({"사업"})
                if "원재료 및 생산설비" in section and not any(topic[0] == "investment" for topic in topics):
                    continue
                if (is_overview or is_product_change) and (_TABLE_LEAD_IN.search(sentence) or _PERFORMANCE_NUMBERS.search(sentence) or _ASPIRATION_STRATEGY.search(sentence)):
                    continue
                if is_product_change and re.search(r"(?:산업\s*전망|시장\s*전망|향후\s*전망)", sentence) and not re.search(r"당사는|당사의|당사가", sentence):
                    continue
                if not multi and not re.search(r"사업\s*(?:구성|흐름)|매출|비중|시장|수요|투자|판매망", question):
                    topics = frozenset(topic for topic in topics if topic[0] == "business")
                if is_product_change and "변화만" in question and not re.search(r"시장|수요|매출|비중|투자", question):
                    topics = frozenset(topic for topic in topics if topic[0] == "business")
                if not topics or unknown_scope or (scopes and (not domains & scopes or (scope_exclusive and domains - scopes))):
                    continue
                if _conflicting_percent(sentence, text):
                    continue
                identity = (citation_token(citation), sentence)
                if identity in seen:
                    continue
                seen.add(identity)
                section_base = 0 if ("사업의 개요" in section or section.strip() == "II. 사업의 내용") else 1
                if section_base == 0 or "주요 제품" in section:
                    is_structure = is_overview or multi
                    if is_structure and _OVERARCHING_ENTITY.search(sentence) and not re.search(r"^(?:아울러|이어서|더불어|그\s*밖에|마지막으로)", sentence):
                        order_tier = -2
                    elif is_structure and _CORE_BUSINESS_STRUCTURE.search(sentence):
                        order_tier = -1
                    else:
                        order_tier = 0
                else:
                    order_tier = section_base
                entries.append(_Sentence(sentence, citation, topics, domains,
                    (order_tier, section, index, text),
                    _qualifiers(text) if any(topic[0] == "composition" for topic in topics) else (),
                    tuple((note, source) for note, source in notices
                          if domains & _domains(note) and note != sentence.lstrip("※* "))[:1]))
        candidates.append(sorted(entries, key=lambda entry: entry.order))

    def render(entry: _Sentence, is_subsequent: bool = False) -> str:
        sentence = naming(entry.text, company)
        if sentence.startswith("당사와"):
            tail = ord(company[-1]) - 0xAC00 if company else 0
            consonant = 0 <= tail < 11172 and tail % 28 != 0
            sentence = company + ("과" if consonant else "와") + sentence[len("당사와"):]
        # Attribution preserves a subsidiary/industry subject instead of
        # relabelling its activities as the reporting parent's own activities.
        if re.search(r"(?<![가-힣])당사(?:의|는|가|와|및|\s)", sentence):
            sentence = (f"{company} 공시 원문: “{entry.text}”" if is_overview
                        else f"{company}의 공시에 따르면, {entry.text}")
        elif company not in sentence:
            sentence = f"{company}의 공시에 따르면, {sentence}"
        elif is_overview and re.search(r"1위|선도|안정적|최고|대표적", sentence) and "공시에 따르면" not in sentence:
            sentence = f"{company}의 공시에 따르면, {sentence}"
        if is_overview and is_subsequent:
            prefix = f"{company}의 공시에 따르면, "
            if sentence.startswith(prefix):
                sentence = sentence[len(prefix):]
        if entry.qualifiers:
            sentence = sentence.rstrip(".") + " (" + "; ".join(entry.qualifiers) + ")."
        body = f"{sentence} {citation_token(entry.citation)}"
        if entry.caveats:
            body += " " + " ".join(f"{note} {citation_token(source)}" for note, source in entry.caveats)
        return body

    item_limit = min((limit.maximum for limit in limits if limit.unit in {"항목", "가지"}), default=12)
    paragraph_limit = min((limit.maximum for limit in limits if limit.unit == "문단"), default=12)
    default_sentences = min(item_limit, paragraph_limit, 6) * len(docs) if multi and limits else 2 * len(docs) if multi else 4 if is_overview else 3
    sentence_limit = min((limit.maximum for limit in limits if limit.unit == "문장"), default=default_sentences)
    explicit_sentences = any(limit.unit == "문장" for limit in limits)
    bullets = any(limit.unit in {"항목", "가지"} for limit in limits)
    if not multi:
        budget = min(sentence_limit, item_limit)
        minimum = max((limit.minimum for limit in limits), default=1)
        selected: list[_Sentence] = []
        covered: set[str] = set()
        selected_segments: list[set[str]] = []
        remaining = list(candidates[0])
        while remaining and len(selected) < budget:
            choice = min(remaining, key=lambda entry: (entry.order[0], -len((entry.domains & scopes) - covered), entry.order[1:]))
            if (len(selected) >= minimum and not scopes - covered
                and max(0, choice.order[0]) > max(0, max(entry.order[0] for entry in selected))):
                break
            if explicit_sentences and sum(1 + len(entry.caveats) for entry in selected) + 1 + len(choice.caveats) > sentence_limit:
                remaining.remove(choice)
                continue
            if not is_overview and len(selected) >= minimum and not choice.domains - covered and not re.search(r"[0-9]", choice.text):
                remaining.remove(choice)
                continue
            choice_segs = set(re.findall(r"([가-힣A-Za-z0-9]+부문)", choice.text))
            if (is_overview and len(choice_segs) >= 2
                and re.search(r"(?:부문으로\s*구성|부문으로\s*구분)", choice.text)
                and any(choice_segs <= prior_segs for prior_segs in selected_segments)):
                remaining.remove(choice)
                continue
            selected.append(choice)
            if choice_segs and re.search(r"(?:부문으로\s*구성|부문으로\s*구분)", choice.text):
                selected_segments.append(choice_segs)
            covered.update(choice.domains)
            remaining.remove(choice)
        if not selected or scopes - covered:
            return _limitation("검색된 사업 본문에서 요청한 사업 범위를 온전히 설명하는 문장을 확인하지 못했습니다.", citations)
        lines = [render(entry, is_subsequent=(i > 0)) for i, entry in enumerate(selected)]
        shortage = any((sum(1 + len(entry.caveats) for entry in selected) if limit.unit == "문장" else len(lines)) < limit.minimum for limit in limits)
        if shortage:
            if len(lines) == budget:
                lines.pop()
            lines.append(_limitation("검색된 사업 본문에서 요청한 수량을 채울 독립적인 설명 문장이 부족합니다.", citations[:1]))
        if bullets:
            return "\n".join("- " + line for line in lines)
        if any(limit.unit == "문단" for limit in limits):
            count = min(paragraph_limit, len(lines))
            paragraphs = [" ".join(lines[i * len(lines) // count:(i + 1) * len(lines) // count]) for i in range(count)]
            return "\n\n".join(paragraphs)
        return ("\n\n" if is_overview and not limits else " ").join(lines)

    common = set.intersection(*(set(topic for entry in entries for topic in entry.topics) for entries in candidates))
    if not common:
        return _limitation("검색된 사업 본문에서 요청한 모든 문서에 대응되는 동일한 사업 주제의 설명을 확인하지 못해 사업 변화의 직접 비교에 한계가 있습니다.", citations)
    budget = min(sentence_limit // len(docs), item_limit, paragraph_limit)
    if budget < 1:
        return _limitation("요청한 분량 안에서 모든 문서의 동일 주제 원문을 의미 손실 없이 제시하기 어렵습니다.", citations)
    composition = bool(re.search(r"사업\s*(?:구성|흐름)", question))
    families = ["composition", "business", "market", "sales", "investment"] if composition else ["business", "composition", "market", "sales", "investment"]
    chosen: list[tuple[Topic, list[_Sentence]]] = []
    used: set[tuple[int, str]] = set()
    pairs = {topic: [next(entry for entry in entries if topic in entry.topics) for entries in candidates] for topic in common}
    def pair_order(topic: Topic) -> tuple:
        pair = pairs[topic]
        sections = {re.sub(r"(?:^|>)\s*[0-9IVX]+\.\s*", ">", str(entry.citation["section"])) for entry in pair}
        domain_key = topic[1].split("/")[0]
        sub_key = topic[1].split("/")[1] if "/" in topic[1] else None
        pat = re.compile(re.escape(sub_key), re.I) if sub_key else _PATTERNS.get(domain_key)
        first_pos = sum(
            min((m.start() for m in [pat.search(entry.text)] if m), default=9999)
            for entry in pair
        ) if pat else 9999
        company_affinity = -1 if topic[1] in company else 0
        return (families.index(topic[0]), company_affinity, max(entry.order[0] for entry in pair),
                first_pos, len(sections), sum(entry.order[2] for entry in pair), topic[1])
    for topic in sorted(common, key=pair_order):
        pair = pairs[topic]
        if explicit_sentences and sum(1 + len(entry.caveats) for _, prior in chosen for entry in prior) + sum(1 + len(entry.caveats) for entry in pair) > sentence_limit:
            continue
        if any((i, entry.text) in used for i, entry in enumerate(pair)):
            continue
        if is_product_change and chosen and any(entry.order[0] > 0 for entry in pair):
            continue
        chosen.append((topic, pair))
        used.update((i, entry.text) for i, entry in enumerate(pair))
        if len(chosen) == budget:
            break
    if not chosen:
        return _limitation("요청한 분량 안에서 동일 주제의 설명과 필요한 공시 단서를 함께 제시하기 어렵습니다.", citations)
    lines = []
    for topic, pair in chosen:
        label = f"{_FAMILIES[topic[0]]}({topic[1].removeprefix('메모리/')})"
        entries = [f"{label} · {_document_label(doc)} 기준: {render(entry)}" for doc, entry in zip(docs, pair)]
        lines.append(("- " if bullets else "") + (" " if bullets else "\n").join(entries))
    shortage = any((sum(1 + len(entry.caveats) for _, pair in chosen for entry in pair) if limit.unit == "문장" else len(chosen)) < limit.minimum for limit in limits)
    if shortage:
        message = _limitation("검색된 사업 본문에서 요청한 수량을 채울 동일 주제의 독립적인 설명이 부족합니다.", citations)
        # A whole aligned topic is the indivisible unit. Never drop one year.
        if len(chosen) * len(docs) >= sentence_limit or (bullets and len(lines) >= item_limit):
            lines.pop()
        lines.append(("- " if bullets else "") + message)
    elif not limits and re.search(r"변화|변동|추이|어떻게\s*달라|어떻게\s*변했", question):
        if len(docs) == 2 and all(re.sub(r"\s+", " ", p[0].text).strip() == re.sub(r"\s+", " ", p[1].text).strip() for _, p in chosen):
            note = "조회된 발췌의 서술이 동일하며, 공시 서술의 일치 또는 일부 서술 차이만으로 회사 전체 사업의 실질적 변경을 입증하는 것은 아닙니다."
            lines.append(("- " if bullets else "") + note)
    return ("\n\n" if any(limit.unit == "문단" for limit in limits) else "\n").join(lines)


__all__ = ["render_quality_narrative"]
