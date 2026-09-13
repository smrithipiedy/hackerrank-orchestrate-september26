import logging
import os
import pandas as pd
from typing import Optional

from code.ingestion import DataIngestor
from code.perception_pipeline import PerceptionPipeline
from code.decision_engine import DecisionEngine
from code.schemas import OutputRow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(dataset_dir: str = "dataset", output_path: str = "dataset/output.csv",
                 api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Executes the complete Buy or Wait? financial decision agent pipeline:
    1. Ingestion: loads profiles, events, rates, requests, options, messages, images.
    2. Perception: resolves image evidence via OCR, parses message amendments, resolves conflicts.
    3. Evaluation: evaluates every request deterministically using cash-flow simulation,
       combinatorial plan optimization, 6-tier tie-breaking, and grounded explanation synthesis.
    4. Validation: validates every output row against the strict Pydantic OutputRow schema.
    5. Output: writes dataset/output.csv matching the required contract.
    """
    logger.info("Initializing Buy or Wait? Financial Decision Agent Pipeline...")

    # Step 1: Ingestion
    logger.info("Step 1: Ingesting dataset files...")
    ingestor = DataIngestor(dataset_path=dataset_dir)
    ingestor.load_all()

    # Step 2: Perception & Conflict Resolution
    logger.info("Step 2: Resolving multimodal evidence and constructing cleaned ledger...")
    pipeline = PerceptionPipeline(ingestor, api_key=api_key)
    cleaned_ledger = pipeline.process_all_users()
    logger.info(f"Cleaned ledger constructed with {len(cleaned_ledger)} verified events.")

    # Step 3: Decision Engine Initialization
    engine = DecisionEngine()
    requests_df = ingestor.requests
    profiles_df = ingestor.profiles
    options_df = ingestor.options

    logger.info(f"Step 3: Evaluating {len(requests_df)} evaluation requests...")
    output_rows = []

    for idx, req_row in requests_df.iterrows():
        req_id = req_row['request_id']
        u_id = req_row['user_id']

        user_profile_series = profiles_df[profiles_df['user_id'] == u_id]
        if user_profile_series.empty:
            logger.warning(f"Profile missing for user {u_id}, request {req_id}")
            user_profile = {'user_id': u_id, 'current_available_balance': 0.0, 'minimum_balance_to_keep': 0.0, 'home_currency': 'USD'}
        else:
            user_profile = user_profile_series.iloc[0].to_dict()

        decision = engine.evaluate_request(
            request_id=req_id,
            request_data=req_row.to_dict(),
            user_profile=user_profile,
            cleaned_ledger=cleaned_ledger,
            payment_options=options_df
        )

        # Step 4: Strict Schema Validation
        try:
            validated_row = OutputRow.model_validate(decision)
            output_rows.append(validated_row.model_dump())
        except Exception as e:
            logger.error(f"Schema validation failed for request {req_id}: {e}")
            output_rows.append(decision)

    # Step 5: Format & Write Output CSV
    out_df = pd.DataFrame(output_rows)
    required_cols = [
        'request_id',
        'amount_safe_to_pay',
        'affordability_status',
        'recommended_payment_method',
        'payment_plan',
        'earliest_date_for_full_payment',
        'spending_changes_needed',
        'decision_explanation'
    ]
    out_df = out_df[required_cols]
    out_df.to_csv(output_path, index=False)
    logger.info(f"Successfully generated {output_path} with {len(out_df)} rows.")

    # Also write to root-level output.csv for submission packaging
    if output_path != "output.csv":
        out_df.to_csv("output.csv", index=False)
        logger.info(f"Successfully generated root output.csv with {len(out_df)} rows.")

    return out_df

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    run_pipeline(dataset_dir="dataset", output_path="dataset/output.csv", api_key=api_key)

if __name__ == "__main__":
    main()
