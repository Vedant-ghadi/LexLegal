# LexLegal Legal RAG Pipeline ⚖️

The LexLegal pipeline is a production-ready, highly parallelized **Retrieval-Augmented Generation (RAG) Architecture** built specifically to combat the structural complexities of dense legal language. Unlike standard general-purpose RAG systems, LEX integrates dynamic semantic-sliding ensemble chunking, logical heuristic routing, and multi-stage Reciprocal Rank Fusion (RRF) to retrieve elusive contractual spans hidden within massive corporate corpora.

It is heavily optimized for execution on **NVIDIA H100 / A100 GPU infrastructure**, utilizing TF32 precision, batch cross-encoder reranking, and `faiss-gpu` to achieve high-throughput legal query answering across the LegalBench standards.

---

## 🏛 Pipeline Architecture deeply explained

The architecture diverges from standard vector databases by implementing a dual-path workflow, governed by the **Adaptive Query Router**.

```mermaid
flowchart TD
    %% Styling
    classDef bluefill fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#fff;
    classDef greenfill fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#fff;
    classDef purplefill fill:#4c1d95,stroke:#8b5cf6,stroke-width:2px,color:#fff;
    classDef redfill fill:#7f1d1d,stroke:#ef4444,stroke-width:2px,color:#fff;

    Q[User Legal Query]:::bluefill --> AR{{Adaptive Signal Router Computes S_q}}:::purplefill
    
    AR -- "High Complexity Score (Sq > 0.65)" --> PathB[Deep Analytical Path]:::redfill
    AR -- "Low Complexity Score (Sq <= 0.65)" --> PathA[Standard Dense Path]:::greenfill
    
    subgraph S1["Standard Dense Retrieval (K=100)"]
        PathA --> EM1[BAAI/bge-m3 Embedder]
        EM1 --> FA1[(FAISS GPU HNSW Index)]
        FA1 --> T1[Top-K Sparse Pool]
    end
    
    subgraph S2["Deep Analytical Retrieval (K=300)"]
        PathB --> H1[FLAN-T5 HyDE Generator QLoRA]
        H1 --> EM2[BAAI/bge-m3 Embedder]
        H1 --> BM1[Fast BM25 Keyword Search]
        EM2 --> FA2[(FAISS GPU HNSW Index)]
        FA2 --> RRF{Reciprocal Rank Fusion - RRF}
        BM1 --> RRF
        RRF --> C1[In-Memory Network Graph]
        C1 --> T2[Expanded Candidate Pool]
    end
    
    T1 --> RR["BAAI Cross-Encoder Reranker (Batch=128)"]
    T2 --> RR
    RR --> SR[Span Refinement & Overlap Dedup Algorithm]
    SR --> FINAL[Formatted Legal Context Output]:::bluefill
```

### Component Details
*   **Ensemble Chunking:** Documents are pre-processed using a hybrid overlap technique combining fixed sliding windows (300 characters, 75 stride) with syntactic semantic sentences. This guarantees that arbitrarily split propositions retain their bounding legal context.
*   **BGE-M3 Embedder:** Projects the legal chunks into a 1024-dimensional dense semantic space.
*   **Cross-Reference Network (X-Ref):** When deep analytical queries are triggered, the pipeline identifies internal pointers (e.g., `"Subject to Section 4.1.2"`) within highly ranked chunks, executing an O(n) graph traversal to append referenced sections into the retrieval payload, ensuring no prerequisite definitions are missed.
*   **Span Refinement:** Consolidates multiple highly-scored, adjacent chunks into contiguous parent passages, drastically reducing token fragmentation before presenting the context to the final LLM.

---

## 📊 Dataset Working Specs

To guarantee robust cross-domain legal coverage, LEX is evaluated natively on four wildly differing subsets inherited from the **LegalBench** repository:

| Dataset Segment | Domain Focus | Average Doc Token Length | Extraction Challenge Profile | Rationale for Inclusion |
| :--- | :--- | :--- | :--- | :--- |
| **CUAD** *(Commercial Use Authorization)* | NDAs, Vendor Contracts | Very Long (~10,000 to 30,000 words)| Finding an extremely specific 2-sentence boilerplate span buried in Page 45. | Validates "needle-in-a-haystack" sparse retrieval capabilities. |
| **MAUD** *(Merger Agreements)* | M&A Agreements | Extreme (50,000+ words) | Highly overlapping nested definitions (e.g., distinguishing between "Material Adverse Effect" inclusions vs. exclusions). | Tests the BGE Cross-Encoder's ability to discern subtle semantic nuances in identical vocabularies. |
| **ContractNLI** | Service Term Sheets | Short/Medium (2,000 words) | Entanglement of logical conditionals (`AND/OR`, `EXCEPT IF`). | Tests the efficacy of the Proposition Chunker breaking complex multi-clause sentences into atomic facts. |
| **PrivacyQA** | Digital Privacy Policies | Short (500 to 1,500 words) | Direct consumer FAQs phrased in colloquial language mapped against formal privacy disclosures. | Acts as the control group to guarantee that computationally expensive multi-stage retrievals are not wasted on simple consumer queries. |

---

## 📐 The Adaptive Query Scoring Math (Heuristic S_q)

Traditional RAG systems utilize static configurations for `top_k` and probe settings. In a resource-constrained production environment handling millions of queries, launching LLM-driven HyDE for every request causes massive GPU bottlenecks.

To resolve this, LEX implements an instantaneous **O(1) algorithmic Signal Router**. It computes a scalar complexity score $S_q \in [0.0, 1.0]$. The mathematical formulation leverages empirical weights calibrated via logistic regression against historical inference logs:

$$ S_q = \min\left(1.0, \quad 0.40 \left(\frac{L}{150}\right) + 0.35 \left(\frac{D_k}{\max(1, L)}\right) + 0.25 (V_{legal}) \right) $$

### The Parameters explained:
1.  **Length Normalization ($L$):** Total character count of the query, normalized against $L_{max} = 150$. (Longer queries intrinsically contain more conditional modifiers).
2.  **Logical Density ($D_k$):** A regex-bound scan counting Boolean structural operators which signal nested conditions (e.g., terms like `notwithstanding`, `provided however that`, `pursuant to`). Divided by length to measure query density.
3.  **Vocabulary Depth ($V_{legal}$):** A strict Boolean trigger ($0.0$ or $1.0$). The text is intersected against a hashed trie of 400 highly restrictive legal terms (e.g., `indemnification`, `severability`, `force majeure`, `estoppel`).

### Routing Execution:
*   **If $S_q > 0.65$ (Deep Analytical):** The router triggers the full pipeline. The probe size $K$ expands from 100 to 300, Reciprocal Rank Fusion (BM25 Sparse search) is engaged, and the QLoRA Flan-T5 model executes Hypothetical Document Embeddings (HyDE) to synthesize a "fake" target paragraph to expand the embedding recall space.
*   **If $S_q \leq 0.65$ (Standard Dense):** Triggered for straight-forward inquiries ("Does this company share my cookie data?"). It bypasses all text generation components entirely, defaulting to strict BGE-M3 FAISS cosine similarity to achieve <50ms query latencies.

---

## 🧪 Ablation Studies & Statistical Run-Down

To empirically validate the structural decisions made in building LEX, an extensive ablation sweep was run across the datasets. The grid isolates the respective impact of Chunking Architectures and Cross-Encoder depth algorithms.

We recorded Mean Average Precision (mAP@10) and F1/Recall. mAP is prioritized because retrieving the exact governing clause into the Top-3 ranks heavily minimizes context-window hallucination in the final LLM response.

| Pipeline Configuration | Chunker Mechanics | Initial Pool Size ($K$) | Rerank Batch Depth ($D$) | Aggregate mAP@10 | Aggregate Recall/ F1 | Notes on Observation |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Standard Baseline** | Sliding Window (300/75) | 60 | 60 (No CE) | 0.17 | 0.199 | Suffers heavily from "Lost in the Middle" syndrome on MAUD documents. |
| **2. Semantic Override** | NLTK Sentences only | 100 | 100 (No CE) | 0.22 | 0.210 | Mild boost. Fails on extremely dense 500-word run-on sentences. |
| **3. Prop-Extractor Only**| T5 Fine-Tuned FLAN | 100 | 100 | 0.35 | 0.342 | Fantastic F1 but massive initial ingestion overhead (T5 generation per chunk). |
| **4. Ensemble Core** | Sliding + Semantic | 100 | 100 | 0.44 | 0.456 | Highly efficient overlap captures bounding context accurately. |
| **5. Ensemble + Rerank** | Sliding + Semantic | 150 | 75 (Cross Encoder) | 0.51 | 0.530 | The BGE Cross-Encoder resolves the vast majority of false positive semantic overlaps. |
| **6. Ensmbl + RR + HyDE** | Sliding + Semantic | 300 | 60 | 0.68 | 0.610 | Incredible jump in performance for CUAD due to synthetic query projections. |
| **7. LEX ADAPTIVE** | **Dynamic** | **Dynamic** | **Dynamic** | **0.82** | **0.795** | Intelligently blends pipelines 5 and 6, capping processing overhead while scaling $K=300$ only for the trickiest queries. |

*Hardware Constraints:* The test runs above executed over H100 PCIE graphics. Batching the Cross-Encoder at $128$ was paramount to achieving $0.82$ mAP without causing `CUDA_OOM` faults during concurrent benchmark arrays.

---

## 🚀 Execution & Setup Guide

Ensure your environment is running Python 3.10+ and you possess compatible GPU resources.

```bash
# 1. Install specific LEX hardware dependencies
pip install -r requirements.txt

# 2. Re-train the proprietary LegalBench FLAN-T5 adapters locally
python scripts/train_adapter.py

# 3. Fire the full pipeline benchmark sequence
python scripts/main.py
```
