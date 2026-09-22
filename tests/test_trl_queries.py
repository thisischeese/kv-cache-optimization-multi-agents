from kv_eval.agents.trl_queries import (
    TRL_EVIDENCE_RULES, TRL_STAGE_CRITERIA,
    build_trl_rag_queries, build_trl_web_queries,
)


def test_all_nine_stages_have_explicit_criteria():
    assert list(TRL_STAGE_CRITERIA) == list(range(1, 10))


def test_research_sources_exclude_survey():
    assert TRL_EVIDENCE_RULES["trl_1_5"]["doc_types"] == ("core", "followup", "benchmark")
    assert len(build_trl_rag_queries("KIVI")) == 4


def test_web_queries_do_not_restrict_domains_or_platforms():
    queries = build_trl_web_queries("KIVI")
    frameworks = [q for q in queries if q["level"] == "trl_6" and q["domains"]]
    assert frameworks == []
    assert all(not q["domains"] for q in queries)
    assert all("KV cache" in q["query"] and "KIVI" in q["query"] for q in queries)
    assert len([q for q in queries if q["level"] == "trl_6" and not q["domains"]]) == 5
    assert {q["source_type"] for q in build_trl_web_queries("KIVI", fallback=True)} == {"framework_doc", "company"}
