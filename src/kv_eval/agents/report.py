"""MOCK report agent. Renders the state into a Markdown document.

Pure string assembly, no LLM call.

TODO: replace the fixed template with an LLM writing pass.
TODO: build REFERENCE from real evidence source ids once retrieval exists.
"""

from kv_eval.schemas import (
    DomainSpec,
    PerspectiveResult,
    Synthesis,
    Tech,
    TechProfile,
    TRLResult,
)
from kv_eval.state import MainState


def _render_bullets(items: list[str]) -> str:
    if not items:
        return "- (none)"
    return "\n".join(f"- {item}" for item in items)


def _render_targets(targets: list[Tech]) -> str:
    if not targets:
        return "- (none)"
    return "\n".join(
        f"- **{tech.name}** (`{tech.tech_id}`, {tech.camp}): {tech.selection_reason}"
        for tech in targets
    )


def _render_profiles(profiles: dict[str, TechProfile]) -> str:
    if not profiles:
        return "(no tech profiles)"

    sections: list[str] = []
    for tech_id, profile in profiles.items():
        sections.append(
            f"### {tech_id}\n\n"
            f"- 개요: {profile.overview}\n"
            f"- 동작 방식: {profile.mechanism}\n"
            f"- 한계:\n{_render_bullets(profile.limitations)}"
        )
    return "\n\n".join(sections)


def _render_perspective(result: PerspectiveResult | None) -> str:
    if result is None:
        return "(missing)"

    evidence = (
        "\n".join(
            f"- `{item.evidence_id}` {item.claim} (source: `{item.source_id}`)"
            for item in result.evidence
        )
        or "- (no evidence)"
    )
    return f"{result.summary}\n\n**Evidence**\n\n{evidence}"


def _render_trl(result: TRLResult | None) -> str:
    if result is None:
        return "(missing)"

    per_tech = (
        "\n".join(f"- **{tech_id}**: {text}" for tech_id, text in result.tech_results.items())
        or "- (no per-tech result)"
    )
    evidence = (
        "\n".join(
            f"- `{item.evidence_id}` {item.claim} (source: `{item.source_id}`)"
            for item in result.evidence
        )
        or "- (no evidence)"
    )
    return f"{per_tech}\n\n{result.summary}\n\n**Evidence**\n\n{evidence}"


def report_agent(state: MainState) -> MainState:
    targets: list[Tech] = state.get("targets", [])
    domain: DomainSpec | None = state.get("domain")
    profiles: dict[str, TechProfile] = state.get("tech_profiles", {})
    synthesis: Synthesis | None = state.get("synthesis")

    problem_definition = domain.problem_definition if domain else "(no domain spec)"
    domain_name = domain.name if domain else "(unknown)"

    matrix_summary = synthesis.matrix_summary if synthesis else "(no synthesis)"
    agreements = _render_bullets(synthesis.agreements if synthesis else [])
    conflicts = _render_bullets(
        [f"{c.topic} — {c.view_a} / {c.view_b}" for c in synthesis.conflicts]
        if synthesis
        else []
    )
    limitations = _render_bullets(synthesis.limitations if synthesis else [])

    report_md = f"""# SUMMARY

> 이 보고서는 1차 스캐폴딩 실행 결과이며, 모든 내용은 MOCK 데이터입니다.

KV cache 최적화 기술 2종(KIVI, InfiniGen)을 `{domain_name}` 도메인에서 네 가지
관점(TRL, 시장성, 이해관계자, 도메인)으로 비교했습니다.

{matrix_summary}

# 1. 분석 배경

{problem_definition}

# 2. 기술 선정

{_render_targets(targets)}

# 3. 기술 개요

{_render_profiles(profiles)}

# 4. 관점별 평가

## 4.1 TRL

{_render_trl(state.get("trl_eval"))}

## 4.2 시장성

{_render_perspective(state.get("market_eval"))}

## 4.3 이해관계자

{_render_perspective(state.get("stakeholder_eval"))}

## 4.4 도메인

{_render_perspective(state.get("domain_eval"))}

# 5. 종합 의견과 시사점

{matrix_summary}

**공통점**

{agreements}

**상충 지점**

{conflicts}

# 6. 한계점

{limitations}

# REFERENCE

TODO: reference builder will populate this section.
"""

    return {"report_md": report_md}
