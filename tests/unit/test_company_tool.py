from disclosure_agent.tools.companies import CompanyResolver


def test_resolves_supplied_aliases_and_rejects_unknown(disclosure_fixture):
    resolver = CompanyResolver(disclosure_fixture["universe"])
    assert resolver.resolve_company("  hyundai-motor co. ")["data"]["corp_code"] == "001"
    assert resolver.resolve_company("005380")["status"] == "ok"
    assert resolver.resolve_company("현대차' OR 1=1 --")["status"] == "not_found"


def test_collision_is_deterministically_ambiguous(disclosure_fixture):
    path = disclosure_fixture["universe"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("003,003003,가상회사,현대차,Virtual,KOSPI,IT,1,IT,,,,,,,,\n")
    result = CompanyResolver(path).resolve_company("현대차")
    assert result["status"] == "ambiguous"
    assert [row["corp_code"] for row in result["data"]] == ["001", "003"]


def test_exact_codes_take_priority_over_text_alias_collision(disclosure_fixture):
    path = disclosure_fixture["universe"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("003,003003,005380,가상회사,Virtual,KOSPI,IT,1,IT,,,,,,,,\n")
    assert CompanyResolver(path).resolve_company("005380")["data"]["corp_code"] == "001"


def test_resolves_note_alias_embedded_in_an_abstract_natural_language_question(
    disclosure_fixture,
):
    path = disclosure_fixture["universe"]
    rows = path.read_text(encoding="utf-8").splitlines()
    rows[1] = rows[1].removesuffix(",") + ",구 현대모터스(2026-04 사명 변경)"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    resolver = CompanyResolver(path)

    resolved = resolver.resolve_company(
        "예전에 현대모터스라는 이름을 사용한 현재 회사를 찾아줘"
    )

    assert resolved["status"] == "ok"
    assert resolved["data"]["corp_code"] == "001"
    assert resolver.resolve_company("현대차' OR 1=1 --")["status"] == "not_found"


def test_embedded_resolution_does_not_hide_a_second_company(disclosure_fixture):
    resolved = CompanyResolver(disclosure_fixture["universe"]).resolve_company(
        "현대차와 SK하이닉스를 비교해줘"
    )

    assert resolved["status"] == "ambiguous"
    assert [row["corp_code"] for row in resolved["data"]] == ["001", "002"]
