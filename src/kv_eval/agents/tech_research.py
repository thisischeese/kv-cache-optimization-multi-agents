"""MOCK tech research agent. Returns both tech profiles in a single pass.

No LLM or retrieval is involved; every value below is hardcoded mock data.

TODO: replace with a per-tech subgraph fanned out from setup:
    setup -> Send(kivi) / Send(infinigen) -> tech_research subgraph
             -> dict reducer -> tech_profiles
Requires a merge reducer on MainState.tech_profiles.
TODO: back the profile fields with RAG over the source papers.
"""

from kv_eval.schemas import TechProfile
from kv_eval.state import MainState

_MOCK_PROFILES: dict[str, TechProfile] = {
    "kivi": TechProfile(
        tech_id="kivi",
        overview=(
            "[MOCK] KIVI is a training-free KV cache quantization method that "
            "pushes the cache toward 2-bit precision to cut serving memory."
        ),
        mechanism=(
            "[MOCK] Applies per-channel quantization to the key cache and "
            "per-token quantization to the value cache, keeping a small "
            "full-precision residual window for recent tokens."
        ),
        limitations=[
            "[MOCK] Quantization error grows on long-context workloads.",
            "[MOCK] Requires custom kernels to realize the theoretical savings.",
        ],
    ),
    "infinigen": TechProfile(
        tech_id="infinigen",
        overview=(
            "[MOCK] InfiniGen is a KV cache management approach for offloading "
            "based LLM inference across the GPU/CPU memory hierarchy."
        ),
        mechanism=(
            "[MOCK] Speculates which KV entries matter for the next attention "
            "step and prefetches only those from CPU memory, overlapping "
            "transfer with compute."
        ),
        limitations=[
            "[MOCK] Depends on host memory bandwidth and PCIe transfer budget.",
            "[MOCK] Speculation misses add latency on irregular attention patterns.",
        ],
    ),
}


def tech_research_agent(state: MainState) -> MainState:
    return {"tech_profiles": dict(_MOCK_PROFILES)}
