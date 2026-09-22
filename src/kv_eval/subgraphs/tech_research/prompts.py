"""Extraction targets and prompt builders for the tech research subgraph.

Queries and extracted text are in English because the source papers are
English; translation belongs to the report stage.
"""

from dataclasses import dataclass

from kv_eval.schemas import Tech
from kv_eval.subgraphs.tech_research.state import RetrievedChunk, SectionPoint

# Keeps prompts bounded while showing every chunk whole. The ingestion splitter
# targets 2200 chars but lets a table or equation stay glued to its sentence up
# to 1.5x that (3300), and a single long element can run past it.
MAX_PASSAGE_CHARS = 4000


@dataclass(frozen=True)
class ItemSpec:
    key: str
    title: str
    instruction: str
    # Words papers actually use for this target; they rarely name it directly.
    vocabulary: str
    seed_query: str  # fallback query, formatted with the tech name


ITEM_SPECS: tuple[ItemSpec, ...] = (
    ItemSpec(
        key="overview",
        title="Overview",
        instruction=(
            "The problem the technique addresses and its core idea, as stated by the authors."
        ),
        vocabulary="problem challenge bottleneck motivation we propose key idea",
        seed_query="{name} KV cache problem motivation core idea",
    ),
    ItemSpec(
        key="mechanism",
        title="Mechanism",
        instruction=(
            "How the technique works: the algorithm, data layout, and key design choices."
        ),
        vocabulary="design algorithm approach procedure step layout implementation",
        seed_query="{name} method algorithm design how it works",
    ),
    ItemSpec(
        key="experiment_setup",
        title="Experiment setup",
        instruction=(
            "The evaluation setting: models, hardware (GPU, CPU, memory, "
            "interconnect), datasets or benchmarks, context length, batch size, "
            "and baselines."
        ),
        vocabulary=(
            "evaluate setup models GPU datasets benchmark batch size sequence length baseline"
        ),
        seed_query="{name} experimental setup models GPU batch size sequence length baselines",
    ),
    ItemSpec(
        key="reported_results",
        title="Reported results",
        instruction=(
            "Quantitative results the authors report (memory reduction, "
            "throughput, latency, accuracy), with the exact numbers and the "
            "setting each number was measured in."
        ),
        vocabulary="achieves speedup throughput latency reduction accuracy table figure",
        seed_query="{name} results throughput speedup memory reduction accuracy",
    ),
    ItemSpec(
        key="limitations",
        title="Limitations",
        instruction=(
            "Limitations, costs, overheads, constraints, assumptions, or failure "
            "cases that the paper itself states. Positive results are not "
            "limitations."
        ),
        vocabulary=(
            "overhead extra memory cost requires trade-off degrade drop bottleneck however only"
        ),
        seed_query="{name} overhead extra memory cost trade-off requires degrade",
    ),
    ItemSpec(
        key="scope",
        title="Scope",
        instruction=(
            "Where the technique applies: supported models or architectures, "
            "hardware requirements, workloads, and integration or compatibility "
            "conditions stated in the paper."
        ),
        vocabulary=(
            "applicable compatible supports requires architecture hardware orthogonal integrate"
        ),
        seed_query="{name} applicable models hardware requirements compatibility",
    ),
    ItemSpec(
        key="competing_views",
        title="Competing approaches",
        instruction=(
            "How the paper characterizes other approaches (for example "
            "quantization, eviction, offloading) and how it compares against "
            "them. Name the exact baseline systems that were compared, and do not "
            "extend a comparison to methods the paper did not evaluate."
        ),
        vocabulary=(
            "compared baselines prior work existing approaches unlike outperforms related work"
        ),
        seed_query="{name} comparison baselines quantization offloading eviction",
    ),
)

ITEM_SPECS_BY_KEY: dict[str, ItemSpec] = {spec.key: spec for spec in ITEM_SPECS}

Messages = list[tuple[str, str]]


def _context(tech: Tech, spec: ItemSpec) -> str:
    return (
        f"Technique: {tech.name}\n"
        f"Extraction target: {spec.title}\n"
        f"Target description: {spec.instruction}"
    )


def _search_context(tech: Tech, spec: ItemSpec) -> str:
    # The team's selection reason helps the LLM write queries, but it is not from the
    # paper, so it stays out of the grade, extract and verify prompts.
    return (
        f"{_context(tech, spec)}\n"
        f"Technique summary (for search only): {tech.selection_reason}\n"
        f"Words the paper may use for this target: {spec.vocabulary}"
    )


def format_passages(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        text = chunk.text[:MAX_PASSAGE_CHARS]
        blocks.append(f"[{number}] ({chunk.doc_id} p.{chunk.page})\n{text}")
    return "\n\n".join(blocks)


QUERY_SYSTEM = (
    "You write one search query for retrieving passages from the research paper "
    "that proposed the given technique. Write it in English, keyword-rich, at "
    "most 20 words, and include the technique name. Use terms that would appear "
    "in the paper's own text, not words like 'paper', 'research', or 'side'."
)

GRADE_SYSTEM = (
    "You judge retrieved passages. A passage is relevant if any sentence in it "
    "fits the target description, even when the rest of the passage is about "
    "something else. It is not relevant if it only mentions the technique, or if "
    "everything in it is outside the target (for example only positive results "
    "when the target is limitations). Return the numbers of the relevant "
    "passages, or an empty list if none are relevant."
)

REWRITE_SYSTEM = (
    "The previous queries did not retrieve passages relevant to the extraction "
    "target. Write one new English search query with different terms. Papers "
    "rarely use the target's name as a word, so describe the concrete content a "
    "passage would contain instead. At most 20 words, and include the technique "
    "name."
)

EXTRACT_SYSTEM = (
    "You extract facts from numbered passages of the paper that proposed the "
    "technique. Use only information in the passages; do not add outside "
    "knowledge. Each point must cite the passage numbers it comes from. Copy "
    "numbers, units, model names, and baseline names exactly. Results are the "
    "authors' own reports; do not present them as independent findings. Every "
    "point must be an instance of the target description; skip passage content "
    "that does not fit it. State only what the passages say: no interpretation "
    "such as 'implying' or 'suggesting', no rounding of numbers, and no claims "
    "about other methods unless the target asks for comparisons. Put passage "
    "numbers only in passage_numbers, never in the text, and do not mention page "
    "numbers. Tables in the passages are flattened text in which a row label such "
    "as a model name may come after its rows, not before; do not attribute a table "
    "number to a specific model, method or metric unless the passage makes that "
    "pairing unambiguous, and prefer numbers stated in sentences. If the passages "
    "do not contain the target information, return an empty list. Write at most 5 "
    "points in English, one or two sentences each."
)


def query_messages(tech: Tech, spec: ItemSpec) -> Messages:
    return [("system", QUERY_SYSTEM), ("human", _search_context(tech, spec))]


def grade_messages(tech: Tech, spec: ItemSpec, chunks: list[RetrievedChunk]) -> Messages:
    human = f"{_context(tech, spec)}\n\nPassages:\n\n{format_passages(chunks)}"
    return [("system", GRADE_SYSTEM), ("human", human)]


def rewrite_messages(
    tech: Tech,
    spec: ItemSpec,
    tried_queries: list[str],
    chunks: list[RetrievedChunk],
) -> Messages:
    tried = "\n".join(f"- {query}" for query in tried_queries)
    seen = format_passages(chunks) if chunks else "(no passages were retrieved)"
    human = (
        f"{_search_context(tech, spec)}\n\nQueries already tried:\n{tried}\n\n"
        f"Passages they returned (judged not relevant):\n\n{seen}"
    )
    return [("system", REWRITE_SYSTEM), ("human", human)]


def extract_messages(tech: Tech, spec: ItemSpec, chunks: list[RetrievedChunk]) -> Messages:
    human = f"{_context(tech, spec)}\n\nPassages:\n\n{format_passages(chunks)}"
    return [("system", EXTRACT_SYSTEM), ("human", human)]


VERIFY_SYSTEM = (
    "You check extracted points against the passages they cite. A point is "
    "supported only if everything it says, including numbers, names, and any "
    "interpretation (for example 'implying', 'does not eliminate'), is stated in "
    "its cited passages. Hedged speculation the passages do not state ('may "
    "imply', 'possibly', 'might') is unsupported. Tables in the passages are "
    "flattened text in which a row label such as a model name may come after its "
    "rows; a point that ties a table number to a specific model, method or metric "
    "is unsupported unless the passage makes that pairing unambiguous. Plain "
    "paraphrase is fine, and so is calling a cost the passage states an overhead. "
    "Give a verdict for every point; "
    "when a point is not supported, name the unsupported part in issue, otherwise "
    "leave issue empty."
)


def verify_messages(
    tech: Tech,
    spec: ItemSpec,
    chunks: list[RetrievedChunk],
    points: list[SectionPoint],
) -> Messages:
    number_by_chunk = {chunk.chunk_id: n for n, chunk in enumerate(chunks, start=1)}
    lines = []
    for n, point in enumerate(points, start=1):
        cited = ", ".join(str(number_by_chunk[c.chunk_id]) for c in point.citations)
        lines.append(f"{n}. {point.text} (cites passages {cited})")
    human = (
        f"{_context(tech, spec)}\n\nPassages:\n\n{format_passages(chunks)}"
        "\n\nPoints:\n" + "\n".join(lines)
    )
    return [("system", VERIFY_SYSTEM), ("human", human)]
