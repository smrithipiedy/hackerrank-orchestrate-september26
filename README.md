# Buy or Wait? Financial Decision Agent

Starter solution and submission for the **HackerRank Orchestrate** 24-hour hackathon (September 2026).

---

## 1. What the Solution Does

The **Buy or Wait?** system is an AI-powered financial decision agent designed to evaluate user purchase requests from `dataset/requests.csv`. For every request, the agent personalizes recommendations based on:
- User financial profiles (`dataset/financial_profiles.csv`), including starting balances, minimum balance limits, protected categories, and installment preferences.
- Historical, scheduled, and pending financial events (`dataset/financial_events.csv`).
- Multimodal evidence from receipts/invoices (`dataset/images.csv`, `dataset/media/images/`) and communication records (`dataset/messages.csv`).
- Fixed dated foreign exchange rates (`dataset/exchange_rates.csv`).
- Available merchant payment options (`dataset/request_payment_options.csv`).

For each request, the agent outputs:
1. `amount_safe_to_pay`: Safe initial payment on the request date.
2. `affordability_status`: `affordable_now`, `affordable_with_plan`, `affordable_later`, or `not_affordable`.
3. `recommended_payment_method`: `full_payment`, `partial_payment`, `installments`, `wait`, or `not_recommended`.
4. `payment_plan`: Chronological schedule formatted as `YYYY-MM-DD:amount|...` or `none`.
5. `earliest_date_for_full_payment`: Conservative projected safe date for a single full payment.
6. `spending_changes_needed`: Up to 3 permitted flexible adjustments (`stop:<id>` or `reduce_to:<id>:<amt>`) or `none`.
7. `decision_explanation`: Concise, grounded explanation of the financial rationale.

---

## 2. Setup and Execution Instructions

### Environment & Python Version
- **Python**: 3.10+ (Developed and tested with Python 3.13.2)
- Platform: Cross-platform (Windows, macOS, Linux)

### Installation
Install project dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Running the Pipeline
Execute the complete end-to-end pipeline:

```bash
python -m code.main
```

This processes all inputs in `dataset/` and outputs the evaluation results to both `dataset/output.csv` and root `output.csv`.

### Verifying the Submission
Validate the output against the challenge schema, row bounds, and packaging requirements:

```bash
python verify_submission.py
```

---

## 3. High-Level Architecture & Approach

The pipeline is organized into modular phases:

1. **Dataset Ingestion (`code/ingestion.py`)**:
   - Ingests all CSV files and maps foreign currencies to user home currencies using fixed dated exchange rates.
2. **Multimodal Evidence Processing & OCR (`code/ocr_vision.py`)**:
   - Recovers missing transaction amounts in raw events by extracting values from receipt and invoice images.
   - Uses local image disk caching (`.cache/ocr_cache.json`) and an offline EasyOCR engine, with optional Gemini 3.8 Flash multimodal integration.
3. **Message Parsing & Conflict Resolution (`code/message_parser.py`, `code/conflict_resolver.py`)**:
   - Extracts date shifts, amount adjustments, and cancellations from user messages.
   - Built-in circuit breaker trips upon any rate-limit (HTTP 429 / `RESOURCE_EXHAUSTED`), gracefully falling back without blocking execution.
   - Resolves conflicts strictly by hierarchy: explicit cancellations/amendments > newer source records > settled status > conservative interpretation.
4. **Deterministic Cash-Flow Simulation (`code/cashflow_simulator.py`)**:
   - Projects daily available balances across the forecast period.
   - Strictly enforces that balances never drop below `minimum_balance_to_keep`.
   - Conservatively handles pending debits (reserved) and unconfirmed credits/gains (excluded until settled).
5. **Combinatorial Plan Optimizer (`code/plan_optimizer.py`)**:
   - Evaluates full payment, partial payment (2-payment schedule with deadline constraint), provider installment plans, and earliest safe wait dates.
   - Exhaustively explores permitted non-protected spending adjustments (stop / reduce) up to the 3-action limit.
6. **6-Tier Decision Engine (`code/decision_engine.py`)**:
   - Evaluates candidate plans using deterministic multi-tier ranking:
     1. Completion by desired completion date
     2. Minimal spending changes needed (0 > 1 > 2 > 3)
     3. Minimal total financial cost
     4. Earliest plan start date
     5. Fewest payment installments
     6. Stable deterministic tie-breaker
7. **Explanation Synthesis (`code/explanation_generator.py`)**:
   - Synthesizes grounded, concise explanations reflecting the exact financial facts, balances, and chosen payment plan.
8. **Schema Validation & Output Generation (`code/schemas.py`, `code/main.py`)**:
   - Validates each row against the strict Pydantic `OutputRow` contract before writing to CSV.

---

## 4. Input / Output Overview

- **Input Directory**: `dataset/`
  - `financial_profiles.csv`, `financial_events.csv`, `exchange_rates.csv`, `requests.csv`, `request_payment_options.csv`, `messages.csv`, `images.csv`, `media/images/`
- **Output Files**:
  - `output.csv` (repository root) and `dataset/output.csv`
  - Exactly 250 rows corresponding to `dataset/requests.csv` with zero schema violations.

---

## 5. Model & API Usage Summary

- **Architecture**: Hybrid neuro-symbolic design with local-first deterministic execution.
- **Model / API Calls**: As documented in `evaluation/usage_report.md`, all financial simulations, optimizations, and decision selections run deterministically with zero runtime LLM costs ($0.00). When running without an API key or when API quota is exhausted, the system automatically uses deterministic message fallback and local OCR.
- **Security & Privacy**: No hardcoded API keys or secrets. Environment variables are loaded securely via `.env` using `python-dotenv` if provided.
