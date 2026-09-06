import pytest

from disclosure_agent.agent.presentation import compact_citations, expand_citations, present_ranking_amounts, strip_verified_amount_annotations

TOKEN = "[근거: ［기재정정］사업보고서 (2024.12) | 20250311001085 | III. 재무에 관한 사항 > 연결재무제표]"

def test_citation_round_trip():
    rendered = compact_citations(TOKEN)
    label, url = rendered.split("](")
    assert "20250311001085" not in label and "…001085" in label
    assert "rcpNo=20250311001085" in url
    assert expand_citations(rendered) == TOKEN
    assert compact_citations(rendered) == rendered

@pytest.mark.parametrize("old,new", [("…001085", "…999999"), ("dart.fss.or.kr", "evil.example"), ("rcpNo=20250311001085", "rcpNo=20250311001086"), ("https://", "http://")])
def test_invalid_link_never_becomes_canonical(old, new):
    invalid = compact_citations(TOKEN).replace(old, new)
    assert expand_citations(invalid) == invalid

@pytest.mark.parametrize("amount,unit,expected", [("575,387", "백만원", "5,753.87"), ("363,304,463,263", "원", "3,633.04"), ("1,234,500", "천원", "12.35"), ("1.2345", "조원", "12,345.00"), ("-575,387", "백만원", "-5,753.87"), ("(575,387)", "백만원", "-5,753.87"), ("△575,387", "백만원", "-5,753.87"), ("0", "원", "0.00")])
def test_conversion(amount, unit, expected):
    source = f"- A 연결 영업이익: {amount}{unit}. {TOKEN}"
    shown = present_ranking_amounts(source)
    assert f"{amount}{unit} (환산 약 {expected}억원)" in shown
    assert strip_verified_amount_annotations(shown) == source
    assert present_ranking_amounts(shown) == shown

def test_unsupported_and_forged_conversion():
    for source in ("100USD", "2.25%", "100억원"):
        assert present_ranking_amounts(source) == source
    bad = "575,387백만원 (환산 약 99.00억원)"
    assert strip_verified_amount_annotations(bad) == bad

def test_inline_bindings_round_trip():
    source = f"2023년: 이전 사업입니다. {TOKEN}\n2024년: 새로운 사업입니다. {TOKEN}"
    assert expand_citations(compact_citations(source)) == source
