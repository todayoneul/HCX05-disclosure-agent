from disclosure_agent.tools.companies import CompanyResolver


def test_resolves_supplied_aliases_and_rejects_unknown(disclosure_fixture):
    resolver = CompanyResolver(disclosure_fixture["universe"])
    assert resolver.resolve_company("  hyundai-motor co. ")["data"]["corp_code"] == "001"
    assert resolver.resolve_company("005380")["status"] == "ok"
    assert resolver.resolve_company("현대차' OR 1=1 --")["status"] == "not_found"


def test_resolves_company_glued_to_following_text_at_a_particle_boundary(
    disclosure_fixture,
):
    # Users often write with no spaces ("카카오의감자결정내역"). The name must still
    # resolve when it is immediately followed by a Korean particle (조사).
    resolver = CompanyResolver(disclosure_fixture["universe"])
    assert resolver.resolve_company("현대차의감자결정내역알려줘")["data"]["corp_code"] == "001"
    assert resolver.resolve_company("SK하이닉스를분석해줘")["data"]["corp_code"] == "002"
    # Safety: a longer distinct token that merely STARTS with an in-universe name
    # but is NOT followed by a particle stays unknown (no false prefix match),
    # exactly like "카카오게임즈" must not collapse to "카카오".
    assert resolver.resolve_company("현대차그룹의매출은얼마야")["status"] == "not_found"


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


def test_resolves_curated_common_market_shorthand_inside_question(disclosure_fixture):
    path = disclosure_fixture["universe"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "003,005930,삼성전자,삼성전자,SAMSUNG ELECTRONICS,KOSPI,IT,1,"
            "반도체·전자부품,,,,,,,,\n"
        )
    resolver = CompanyResolver(path)

    assert resolver.resolve_company("삼전 23년 연결 매출")["data"]["corp_code"] == "003"
    assert resolver.resolve_company("하닉 3분기 실적")["data"]["corp_code"] == "002"


def test_resolves_curated_historical_name_only_when_successor_is_in_universe(
    disclosure_fixture,
):
    path = disclosure_fixture["universe"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "003,028050,삼성E&A,삼성E&A,SAMSUNG E&A,KOSPI,산업재,16,"
            "건설,,,,,,,,\n"
        )

    resolved = CompanyResolver(path).resolve_company(
        "삼성엔지니어링의 2024년 사업보고서 회사의 개요를 알려줘"
    )

    assert resolved["status"] == "ok"
    assert resolved["data"]["corp_code"] == "003"


def test_resolves_every_name_form_including_embedded_in_a_question(disclosure_fixture):
    # Official / listed / stock code / English (raw + suffix-stripped) / Hangul
    # initialism phonetic / curated shorthand — bare and inside a question.
    # Fixture: 001 현대자동차(현대차, "HYUNDAI MOTOR CO"), 002 SK하이닉스("SK hynix Inc.").
    resolver = CompanyResolver(disclosure_fixture["universe"])

    def code(q: str) -> str:
        r = resolver.resolve_company(q)
        assert r["status"] == "ok", (q, r["status"])
        return r["data"]["corp_code"]

    assert code("현대차") == "001"
    assert code("005380") == "001"
    assert code("hyundai motor") == "001"  # English suffix stripped
    assert code("HYUNDAI MOTOR CO의 2024년 매출은?") == "001"
    assert code("SK hynix") == "002"
    assert code("SK hynix Inc.의 2024년 연결 매출액은 얼마인가요?") == "002"
    assert code("에스케이하이닉스") == "002"  # Hangul initialism phonetic
    assert code("에스케이하이닉스의 2024년 매출") == "002"
    assert code("하이닉스 3분기 실적") == "002"  # curated shorthand


def test_embedded_match_rejects_a_longer_distinct_company_name(disclosure_fixture):
    # "카카오게임즈" is a different company than the in-universe "카카오"; a prefix
    # substring must not resolve it (that would serve one company's data for
    # another). The genuine "카카오" reference must still resolve.
    path = disclosure_fixture["universe"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "003,035720,카카오,카카오,KAKAO,KOSPI,IT,1,플랫폼,,,,,,,,\n"
        )
    resolver = CompanyResolver(path)

    assert (
        resolver.resolve_company("카카오게임즈의 2024년 연결 매출액은 얼마인가요?")["status"]
        == "not_found"
    )
    assert (
        resolver.resolve_company("카카오의 2024년 연결 매출액은?")["data"]["corp_code"]
        == "003"
    )
    assert (
        resolver.resolve_company("카카오 2024년 매출")["data"]["corp_code"] == "003"
    )


def test_resolves_contest_name_forms_and_fails_closed_on_group_only_name(
    disclosure_fixture,
):
    path = disclosure_fixture["universe"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "003,000270,기아,기아,KIA CORPORATION,KOSPI,자동차,1,자동차,,,,,,,,\n"
            "004,005490,POSCO홀딩스,POSCO홀딩스,POSCO HOLDINGS INC.,KOSPI,철강,1,철강,,,,,,,,\n"
            "005,005930,삼성전자,삼성전자,SAMSUNG ELECTRONICS CO., LTD.,KOSPI,IT,1,반도체,,,,,,,,\n"
            "006,373220,LG에너지솔루션,LG에너지솔루션,LG ENERGY SOLUTION LTD.,KOSPI,배터리,1,배터리,,,,,,,,\n"
            "007,006400,삼성SDI,삼성SDI,SAMSUNG SDI CO., LTD.,KOSPI,배터리,1,배터리,,,,,,,,\n"
        )
    resolver = CompanyResolver(path)

    expected = {
        "기아의 2024년 매출": "003",
        "기아자동차 2024년 매출": "003",
        "Kia의 2024년 매출": "003",
        "POSCO홀딩스 2024년 매출": "004",
        "포스코의 2024년 매출": "004",
        "Samsung Electronics 2024년 매출": "005",
        "Samsung Electronics' 2024 consolidated revenue": "005",
        "005930번 종목의 2024년 매출": "005",
        "엘지에너지솔루션 2024년 매출": "006",
        "하이닉스 2024년 매출": "002",
        "에스케이하이닉스 2024년 매출": "002",
    }
    for query, corp_code in expected.items():
        resolved = resolver.resolve_company(query)
        assert resolved["status"] == "ok", (query, resolved)
        assert resolved["data"]["corp_code"] == corp_code

    assert resolver.resolve_company("삼성의 2024년 매출")["status"] != "ok"


def test_resolves_common_hanwha_aerospace_shorthand(disclosure_fixture):
    path = disclosure_fixture["universe"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "003,012450,한화에어로스페이스,한화에어로스페이스,"
            "HANWHA AEROSPACE CO., LTD.,KOSPI,산업재,14,방산,,,,,,,,\n"
        )

    resolved = CompanyResolver(path).resolve_company(
        "한화에어로의2024년사업보고서주요사업을요약해줘"
    )

    assert resolved["status"] == "ok"
    assert resolved["data"]["corp_code"] == "003"


def test_resolves_multiple_embedded_names_in_a_no_space_mixed_script_question(
    disclosure_fixture,
):
    path = disclosure_fixture["universe"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "003,005930,삼성전자,삼성전자,SAMSUNG ELECTRONICS CO., LTD.,"
            "KOSPI,IT,1,반도체,,,,,,,,\n"
        )
    resolver = CompanyResolver(path)

    resolved = resolver.resolve_company(
        "CompareSamsungElectronics와에스케이하이닉스의2024년연결당기순이익"
    )

    assert resolved["status"] == "ambiguous"
    assert [row["corp_code"] for row in resolved["data"]] == ["002", "003"]


def test_resolves_english_name_glued_to_a_disclosure_document_marker(
    disclosure_fixture,
):
    path = disclosure_fixture["universe"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "003,005930,삼성전자,삼성전자,SAMSUNG ELECTRONICS CO., LTD.,"
            "KOSPI,IT,1,반도체,,,,,,,,\n"
            "004,000270,기아,기아,KIA CORPORATION,KOSPI,자동차,1,자동차,,,,,,,,\n"
        )
    resolver = CompanyResolver(path)

    resolved = resolver.resolve_company(
        "2024년에제출된SamsungElectronics사업보고서의연결매출액"
    )

    assert resolved["status"] == "ok"
    assert resolved["data"]["corp_code"] == "003"
    resolved_korean = resolver.resolve_company(
        "2025년에제출된기아사업보고서의연결매출액"
    )
    assert resolved_korean["status"] == "ok"
    assert resolved_korean["data"]["corp_code"] == "004"
    # Do not turn an unrelated longer English word into the short Kia alias.
    assert resolver.resolve_company("Nokia사업보고서의매출액")["status"] == "not_found"


def test_resolves_no_space_company_names_at_strong_metric_boundaries(
    disclosure_fixture,
):
    path = disclosure_fixture["universe"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "003,005930,삼성전자,삼성전자,SAMSUNG ELECTRONICS CO., LTD.,"
            "KOSPI,IT,1,반도체,,,,,,,,\n"
        )
    resolver = CompanyResolver(path)

    single = resolver.resolve_company("2024년삼성전자매출액은얼마야")
    comparison = resolver.resolve_company(
        "삼성전자와SK하이닉스매출액비교"
    )
    english = resolver.resolve_company(
        "PleasecompareKiaandSamsungElectronicsrevenue"
    )

    assert single["status"] == "ok"
    assert single["data"]["corp_code"] == "003"
    assert comparison["status"] == "ambiguous"
    assert [row["corp_code"] for row in comparison["data"]] == ["002", "003"]
    # The fixture has no Kia row, so only Samsung should resolve here.  The
    # leading prose and glued English metric must still form safe boundaries.
    assert english["status"] == "ok"
    assert english["data"]["corp_code"] == "003"


def test_metric_boundary_scan_does_not_collapse_longer_unrelated_names(
    disclosure_fixture,
):
    path = disclosure_fixture["universe"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "003,035720,카카오,카카오,KAKAO,KOSPI,IT,1,플랫폼,,,,,,,,\n"
            "004,000270,기아,기아,KIA CORPORATION,KOSPI,자동차,1,자동차,,,,,,,,\n"
        )
    resolver = CompanyResolver(path)

    assert resolver.resolve_company("카카오게임즈매출액")["status"] == "not_found"
    assert resolver.resolve_company("Nokiarevenue")["status"] == "not_found"


def test_resolves_sector_to_every_corpus_candidate_in_source_order(
    disclosure_fixture,
):
    path = disclosure_fixture["universe"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "003,005930,삼성전자,삼성전자,SAMSUNG ELECTRONICS,KOSPI,IT,1,"
            "반도체·전자부품,,,,,,,,\n"
            "004,009150,삼성전기,삼성전기,SAMSUNG ELECTRO-MECHANICS,KOSPI,IT,1,"
            "반도체·전자부품,,,,,,,,\n"
        )
    resolver = CompanyResolver(path)

    resolved = resolver.resolve_sector("2024년 반도체 회사 중 연결 매출 1위는?")

    assert resolved["status"] == "ok"
    assert resolved["data"]["sector"] == "반도체·전자부품"
    assert [row["corp_code"] for row in resolved["data"]["candidates"]] == [
        "002",
        "003",
        "004",
    ]
    assert all(
        set(row) == {"corp_code", "stock_code", "corp_name", "listed_name", "sector"}
        for row in resolved["data"]["candidates"]
    )


def test_sector_resolution_fails_closed_for_mixed_or_unknown_sector(
    disclosure_fixture,
):
    resolver = CompanyResolver(disclosure_fixture["universe"])

    mixed = resolver.resolve_sector("자동차와 반도체 회사 중 1위는?")
    unknown = resolver.resolve_sector("우주광산 회사 중 1위는?")

    assert mixed["status"] == "ambiguous"
    assert [row["sector"] for row in mixed["data"]] == [
        "자동차·모빌리티",
        "반도체·전자부품",
    ]
    assert unknown["status"] == "not_found"


def test_sector_resolution_does_not_match_sector_inside_company_name(
    disclosure_fixture,
):
    path = disclosure_fixture["universe"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "003,293490,카카오게임즈,카카오게임즈,KAKAO GAMES,KOSDAQ,게임,3,"
            "게임,,,,,,,,\n"
        )
    resolver = CompanyResolver(path)

    resolved = resolver.resolve_sector("카카오게임즈의 2024년 매출액은?")

    assert resolved["status"] == "not_found"


def test_sector_resolution_accepts_the_sekteo_group_boundary(disclosure_fixture):
    resolver = CompanyResolver(disclosure_fixture["universe"])

    resolved = resolver.resolve_sector("2024년 반도체 섹터 중 연결 매출 1위는?")

    assert resolved["status"] == "ok"
    assert resolved["data"]["sector"] == "반도체·전자부품"


def test_resolves_english_name_glued_to_a_safe_question_action(
    disclosure_fixture,
):
    path = disclosure_fixture["universe"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "003,005930,삼성전자,삼성전자,SAMSUNG ELECTRONICS CO., LTD.,"
            "KOSPI,IT,1,반도체,,,,,,,,\n"
        )
    resolver = CompanyResolver(path)

    resolved = resolver.resolve_company(
        "UsingonlyDART,summarizeSamsungElectronics'2024businessoverview"
    )

    assert resolved["status"] == "ok"
    assert resolved["data"]["corp_code"] == "003"
    assert resolver.resolve_company("NotSamsungElectronicsrevenue")["status"] == "not_found"
