import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import date
from code.ingestion import DataIngestor
from code.ocr_vision import OCRVision
from code.message_parser import MessageParser

logger = logging.getLogger(__name__)

from code.schemas import CleanedLedgerEntry

class PerceptionPipeline:
    """
    Orchestrates the multimodal evidence pipeline for Phase 2.
    Flow: Events -> (Messages/Images) -> OCR/NLP -> FX Normalization -> Resolved Ledger.
    """

    def __init__(self, ingestor: DataIngestor, api_key: Optional[str] = None):
        self.ingestor = ingestor
        self.ocr = OCRVision(api_key=api_key)
        self.parser = MessageParser(api_key=api_key)
        # Pre-process rates dates for efficiency
        self.ingestor.rates['rate_date'] = pd.to_datetime(self.ingestor.rates['rate_date']).dt.date

    def process_all_users(self) -> pd.DataFrame:
        """Processes all users in the dataset and returns a unified resolved ledger."""
        all_resolved = []

        # Get all users from profiles
        users = self.ingestor.profiles['user_id'].unique()

        # Normalize events once to get home currencies and baseline amounts
        normalized_events = self.ingestor.normalize_events()

        for user_id in users:
            user_events = self.ingestor.get_events_for_user(user_id, normalized_events).copy()

            # 1. Image-based Amount Recovery
            # Find events with blank amounts in the ORIGINAL dataset
            # We look at self.ingestor.events because normalized_events might have NaNs where OCR is needed
            raw_events = self.ingestor.events[self.ingestor.events['user_id'] == user_id]
            blank_events = raw_events[raw_events['amount'].isna()]

            for _, raw_event in blank_events.iterrows():
                eid = raw_event['event_id']
                # Link event_id to image_id in images.csv
                image_row = self.ingestor.images[self.ingestor.images['related_event_id'] == eid]
                if not image_row.empty:
                    image_id = image_row.iloc[0]['image_id']
                    image_path = f"dataset/media/images/{image_id}.png"

                    # Extract amount and currency from image
                    amount, currency = self.ocr.extract_amount(image_path)
                    if amount is not None:
                        # Define event mask for lookups
                        event_mask = (user_events['event_id'] == eid)
                        # Fallback to event currency if OCR returns "Unknown"
                        if currency == "Unknown":
                            currency = user_events.loc[event_mask, 'currency'].iloc[0]

                        # Retrieve user profile for home currency
                        profile = self.ingestor.get_user_profile(user_id)
                        if profile:
                            home_curr = profile['home_currency']
                            # Get settlement date from the event
                            event_mask = (user_events['event_id'] == eid)
                            settlement_date = pd.to_datetime(user_events.loc[event_mask, 'settlement_date'].iloc[0]).date()

                            # Normalize extracted foreign currency to home currency using dated FX rate
                            normalized_amount = self._normalize_value(amount, currency, home_curr, settlement_date)

                            # Update the ledger using boolean masking
                            user_events.loc[event_mask, 'amount_home'] = normalized_amount
                            user_events.loc[event_mask, 'amount'] = amount # Store recovered original amount

            # 2. Message-based Amendments
            # Filter messages for this user
            user_messages = self.ingestor.messages[self.ingestor.messages['user_id'] == user_id]
            amendments = []
            for _, msg in user_messages.iterrows():
                context = {
                    'user_id': user_id,
                    'related_event_id': msg.get('related_event_id')
                }
                extracted = self.parser.parse_message(msg['message_text'], context)
                amendments.extend([a.model_dump() for a in extracted])

            # 3. Conflict Resolution
            from code.conflict_resolver import ConflictResolver
            resolver = ConflictResolver()
            resolved_user_ledger = resolver.resolve(user_events, amendments)

            # SCHEMA VALIDATION: Ensure every resolved record matches CleanedLedgerEntry
            validated_records = []
            for _, row in resolved_user_ledger.iterrows():
                try:
                    # Convert row to dict and validate
                    record_dict = row.to_dict()
                    # Handle NaNs for Pydantic (which doesn't like NaN for floats unless Optional)
                    # We replace NaN with None for validation purposes
                    clean_dict = {k: (v if pd.notna(v) else None) for k, v in record_dict.items()}
                    validated_record = CleanedLedgerEntry.model_validate(clean_dict)
                    validated_records.append(validated_record.model_dump())
                except Exception as e:
                    logger.error(f"Schema validation failed for event {row.get('event_id')}: {e}")
                    # If validation fails, we keep the record but log it, or we could drop it.
                    # To be safe and deterministic, we keep the original row but as a dict.
                    validated_records.append(row.to_dict())

            all_resolved.append(pd.DataFrame(validated_records))

        if not all_resolved:
            return pd.DataFrame()

        return pd.concat(all_resolved, ignore_index=True)

    def _normalize_value(self, amount: float, from_curr: str, to_curr: str, date_val: date) -> float:
        """Helper to normalize a single value using the exchange_rates.csv."""
        if from_curr == to_curr:
            return amount

        # Match rate_date, from_currency, to_currency
        # Rates are in self.ingestor.rates
        rates = self.ingestor.rates
        # Ensure date types match
        rates['rate_date'] = pd.to_datetime(rates['rate_date']).dt.date

        match = rates[
            (rates['rate_date'] == date_val) &
            (rates['from_currency'] == from_curr) &
            (rates['to_currency'] == to_curr)
        ]

        if not match.empty:
            return amount * float(match.iloc[0]['rate'])

        logger.warning(f"No exchange rate found for {from_curr}->{to_curr} on {date_val}")
        return np.nan
