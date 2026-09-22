from kv_eval.agents.trl_queries import (
    TRL_RAG_DOC_TYPES,
    build_trl_rag_queries,
    build_trl_web_queries,
    get_document_purpose,
    get_trl_documents,
)
from kv_eval.rag.types import DocumentRecord


def _documents() -> list[DocumentRecord]:
    return [
        DocumentRecord(
            doc_id="kivi",
            file="kivi.pdf",
            title="KIVI",
            tech_id="kivi",
            camp="SW",
            doc_type="core",
        ),
        DocumentRecord(
            doc_id="kvtuner",
            file="kvtuner.pdf",
            title="KVTuner",
            tech_id="kivi",
            camp="SW",
            doc_type="followup",
        ),
        DocumentRecord(
            doc_id="bench_kivi",
            file="bench_kivi.pdf",
            title="KIVI Benchmark",
            tech_id="kivi",
            camp="SW",
            doc_type="benchmark",
        ),
        DocumentRecord(
            doc_id="survey_tmlr",
            file="survey.pdf",
            title="KV Cache Survey",
            tech_id="common",
            camp="common",
            doc_type="survey",
        ),
        DocumentRecord(
            doc_id="infinigen",
            file="infinigen.pdf",
            title="InfiniGen",
            tech_id="infinigen",
            camp="HW",
            doc_type="core",
        ),
    ]


def test_trl_rag_doc_types_exclude_surveys() -> None:
    assert TRL_RAG_DOC_TYPES == [
        "core",
        "followup",
        "benchmark",
    ]
    assert "survey" not in TRL_RAG_DOC_TYPES


def test_build_trl_rag_queries_covers_research_evidence() -> None:
    queries = build_trl_rag_queries("KIVI")

    assert len(queries) == 4
    assert any("technical principle" in query for query in queries)
    assert any("reported results" in query for query in queries)
    assert any("limitations" in query for query in queries)
    assert any("external benchmark" in query for query in queries)


def test_build_trl_web_queries_separates_trl_6_and_trl_7_to_9() -> None:
    queries = build_trl_web_queries("KIVI")

    trl_6_queries = [
        query for query in queries if query["level"] == "trl_6"
    ]
    trl_7_to_9_queries = [
        query for query in queries if query["level"] == "trl_7_9"
    ]

    assert len(trl_6_queries) == 3
    assert len(trl_7_to_9_queries) == 1

    assert {
        query["domains"][0]
        for query in trl_6_queries
    } == {
        "docs.vllm.ai",
        "docs.sglang.ai",
        "nvidia.github.io",
    }

    assert trl_7_to_9_queries[0]["domains"] == []
    assert all("KV cache" in query["query"] for query in queries)
    assert all("LLM inference" in query["query"] for query in queries)


def test_get_trl_documents_includes_common_context() -> None:
    documents = get_trl_documents(_documents(), "KIVI")

    assert {document.doc_id for document in documents} == {
        "kivi",
        "kvtuner",
        "bench_kivi",
        "survey_tmlr",
    }


def test_get_trl_documents_excludes_other_technology() -> None:
    documents = get_trl_documents(_documents(), "kivi")

    assert all(document.tech_id in {"kivi", "common"} for document in documents)
    assert "infinigen" not in {
        document.doc_id for document in documents
    }


def test_get_document_purpose_uses_doc_type() -> None:
    document = _documents()[1]

    assert document.doc_type == "followup"
    assert "후속 개선" in get_document_purpose(document)
