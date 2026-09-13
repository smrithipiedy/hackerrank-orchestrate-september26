# Token Usage and Cost Analysis Report
**Challenge**: HackerRank Orchestrate (September 2026) — Buy or Wait?  
**Run Type**: Final Full-Dataset Execution (`dataset/requests.csv`, 250 requests)  
**Run Date**: 2026-09-13  
**Status**: Completed  

---

## 1. Executive Summary

This report documents the AI and model calls, input and output token usage, and associated costs for the final full-dataset evaluation run of the **Buy or Wait?** financial decision agent, as mandated by `problem_statement.md` and `AGENTS.md` §6.5.

The solution implements a deterministic, multi-tiered architecture:
1. **Multimodal Perception & Amount Recovery**: An OCR vision module (`code/ocr_vision.py`) processes receipt and bill images (`dataset/media/images/`) for blank transaction amounts, supported by local disk caching (`.cache/ocr_cache.json`) and an offline EasyOCR engine.
2. **Deterministic Financial Simulation & Optimization**: All 90-day day-by-day cash-flow forecasting (`code/cashflow_simulator.py`), exhaustive combinatorial spending adjustments (`code/plan_optimizer.py`), 6-tier deterministic plan ranking (`code/decision_engine.py`), and grounded explanation generation (`code/explanation_generator.py`) execute via **100% deterministic Python and NumPy algorithms**, requiring **zero runtime LLM or API calls**.

---

## 2. Model Calls and Token Usage Breakdown

| Category | Model Provider | Model Name | Model Calls | Input Tokens | Output Tokens | Total Tokens | Avg Tokens / Request | Total Cost (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Multimodal OCR (Receipts/Bills)** | Local / EasyOCR | CRNN / ResNet | 16 | 0 | 0 | 0 | 0 | $0.00 |
| **Message Parser (NLP)** | Rule-Based Regex | Deterministic | 0 | 0 | 0 | 0 | 0 | $0.00 |
| **Cash-Flow Simulation** | N/A (Python/NumPy) | Deterministic | 0 | 0 | 0 | 0 | 0 | $0.00 |
| **Plan Optimizer (Combinatorial)** | N/A (Python) | Deterministic | 0 | 0 | 0 | 0 | 0 | $0.00 |
| **6-Tier Decision Engine** | N/A (Python) | Deterministic | 0 | 0 | 0 | 0 | 0 | $0.00 |
| **Explanation Synthesis** | N/A (Python Templates) | Deterministic | 0 | 0 | 0 | 0 | 0 | $0.00 |
| **Total / Overall** | **Deterministic / Local** | — | **16** | **0** | **0** | **0** | **0** | **$0.00** |

*Note on Available Metrics*:
- In compliance with challenge constraints that prohibit live external API requirements during evaluation, the full-dataset run was completed using the deterministic fallback and local vision caching.
- If executed with cloud-hosted Gemini 1.5 Flash via `GEMINI_API_KEY`:
  - Image OCR: 16 image calls $\times \approx 258$ tokens/image $\approx 4,128$ tokens.
  - Estimated Cloud Cost: $< \$0.001$ ($0.00 \text{ USD}$ rounded).
  - Average tokens per evaluation request across all 250 requests: $\approx 16.5$ tokens/request.

---

## 3. Cost Analysis

- **Estimated Total Cost**: **$0.00**
- **Estimated Cost per Request**: **$0.0000**
- **External Network / Live FX Dependencies**: **None** (100% compliant with challenge contract).
