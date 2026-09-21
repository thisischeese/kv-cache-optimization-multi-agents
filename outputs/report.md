# SUMMARY

> 이 보고서는 1차 스캐폴딩 실행 결과이며, 모든 내용은 MOCK 데이터입니다.

KV cache 최적화 기술 2종(KIVI, InfiniGen)을 `cloud_llm_serving` 도메인에서 네 가지
관점(TRL, 시장성, 이해관계자, 도메인)으로 비교했습니다.

[MOCK] KIVI and InfiniGen are compared across 4 perspectives (TRL, Market, Stakeholder, Domain). KIVI reduces the per-token cache footprint; InfiniGen extends effective cache capacity across the memory hierarchy.

# 1. 분석 배경

In cloud LLM serving, KV cache memory growth with batch size and context length limits throughput and raises serving cost.

# 2. 기술 선정

- **KIVI** (`kivi`, SW): Representative training-free KV cache quantization method on the software side.
- **InfiniGen** (`infinigen`, HW): Representative memory-hierarchy/offloading approach on the hardware-system side.

# 3. 기술 개요

### kivi

- 개요: [MOCK] KIVI is a training-free KV cache quantization method that pushes the cache toward 2-bit precision to cut serving memory.
- 동작 방식: [MOCK] Applies per-channel quantization to the key cache and per-token quantization to the value cache, keeping a small full-precision residual window for recent tokens.
- 한계:
- [MOCK] Quantization error grows on long-context workloads.
- [MOCK] Requires custom kernels to realize the theoretical savings.

### infinigen

- 개요: [MOCK] InfiniGen is a KV cache management approach for offloading based LLM inference across the GPU/CPU memory hierarchy.
- 동작 방식: [MOCK] Speculates which KV entries matter for the next attention step and prefetches only those from CPU memory, overlapping transfer with compute.
- 한계:
- [MOCK] Depends on host memory bandwidth and PCIe transfer budget.
- [MOCK] Speculation misses add latency on irregular attention patterns.

# 4. 관점별 평가

## 4.1 TRL

- **kivi**: [MOCK] TRL 4-5. Validated in research prototypes with published kernels, not yet standard in production serving stacks.
- **infinigen**: [MOCK] TRL 4. Demonstrated on offloading-based inference testbeds; integration with mainstream serving engines is early.

[MOCK] Both techniques sit at the prototype-validation stage, with KIVI slightly ahead on ecosystem adoption.

**Evidence**

- `trl-kivi-001` [MOCK] KIVI is evaluated as a training-free drop-in method. (source: `mock-source-kivi`)
- `trl-infinigen-001` [MOCK] InfiniGen is evaluated on offloading-based inference. (source: `mock-source-infinigen`)

## 4.2 시장성

[MOCK] Demand for KV cache reduction is driven by serving cost per token. KIVI targets memory footprint directly, while InfiniGen targets capacity expansion on cheaper memory tiers.

**Evidence**

- `market-001` [MOCK] GPU memory is the binding cost constraint in LLM serving. (source: `mock-source-market`)

## 4.3 이해관계자

[MOCK] Serving operators care about throughput per GPU, model engineers care about accuracy regression, and infrastructure teams care about how much of the stack must be modified.

**Evidence**

- `stakeholder-001` [MOCK] Accuracy regression is the primary adoption blocker for cache compression. (source: `mock-source-stakeholder`)

## 4.4 도메인

[MOCK] In cloud LLM serving, KIVI mainly relaxes the memory capacity ceiling for larger batches, while InfiniGen mainly shifts the bottleneck from GPU capacity to host transfer bandwidth.

**Evidence**

- `domain-001` [MOCK] Batch size and context length jointly drive KV cache growth in multi-tenant serving. (source: `mock-source-domain`)

# 5. 종합 의견과 시사점

[MOCK] KIVI and InfiniGen are compared across 4 perspectives (TRL, Market, Stakeholder, Domain). KIVI reduces the per-token cache footprint; InfiniGen extends effective cache capacity across the memory hierarchy.

**공통점**

- [MOCK] Both target the KV cache as the dominant serving memory cost.
- [MOCK] Both are at prototype maturity rather than production default.

**상충 지점**

- [MOCK] The two shift the bottleneck differently: KIVI trades numerical precision, InfiniGen trades transfer bandwidth.

# 6. 한계점

- [MOCK] All findings in this run are mock data, not retrieved evidence.
- [MOCK] No quantitative benchmark comparison has been performed.

# REFERENCE

TODO: reference builder will populate this section.
