import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import logging

# Setup logging for auditability
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DataIngestor:
    """
    Handles loading and normalization of the financial datasets.
    Ensures all monetary values are converted to the user's home currency
    using the correct dated exchange rates.
    """

    def __init__(self, dataset_path: str = "dataset"):
        self.dataset_path = dataset_path
        self.profiles = None
        self.events = None
        self.rates = None
        self.requests = None
        self.options = None
        self.messages = None
        self.images = None

    def load_all(self):
        """Loads all required CSV files from the dataset directory."""
        try:
            self.profiles = pd.read_csv(f"{self.dataset_path}/financial_profiles.csv")
            self.events = pd.read_csv(f"{self.dataset_path}/financial_events.csv")
            self.rates = pd.read_csv(f"{self.dataset_path}/exchange_rates.csv")
            self.requests = pd.read_csv(f"{self.dataset_path}/requests.csv")
            self.options = pd.read_csv(f"{self.dataset_path}/request_payment_options.csv")
            self.messages = pd.read_csv(f"{self.dataset_path}/messages.csv")
            self.images = pd.read_csv(f"{self.dataset_path}/images.csv")
            logger.info("Successfully loaded all dataset files.")
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            raise e

    def _parse_date(self, date_str: Any):
        """Helper to consistently parse dates."""
        if pd.isna(date_str) or date_str == "":
            return None
        try:
            return pd.to_datetime(date_str).date()
        except Exception:
            return None

    def normalize_events(self) -> pd.DataFrame:
        """
        Normalizes all financial events to the user's home currency.
        Follows the problem statement: use settlement_date for exchange rates.
        """
        if self.profiles is None or self.events is None or self.rates is None:
            raise ValueError("Datasets not loaded. Call load_all() first.")

        # Convert dates to datetime.date objects for consistency
        self.profiles['user_id'] = self.profiles['user_id'].astype(str)
        self.events['user_id'] = self.events['user_id'].astype(str)
        self.rates['rate_date'] = self.rates['rate_date'].apply(self._parse_date)
        self.events['settlement_date'] = self.events['settlement_date'].apply(self._parse_date)

        # Create a lookup for exchange rates: (date, from, to) -> rate
        rate_lookup = {
            (row.rate_date, row.from_currency, row.to_currency): row.rate
            for row in self.rates.itertuples()
        }

        # Merge events with profiles to get home_currency
        merged = self.events.merge(self.profiles[['user_id', 'home_currency']], on='user_id', how='left')

        def get_normalized_amount(row):
            amount = row['amount']
            currency = row['currency']
            home_curr = row['home_currency']
            date = row['settlement_date']

            if pd.isna(amount):
                return np.nan

            if currency == home_curr:
                return float(amount)

            # Look up exact dated rate
            rate = rate_lookup.get((date, currency, home_curr))

            if rate is not None:
                return float(amount) * float(rate)

            # The problem statement says "do not invent fallbacks".
            # If a rate is missing for a specific date, we flag it.
            logger.warning(f"Missing exchange rate for {currency}->{home_curr} on {date}. Event ID: {row['event_id']}")
            return np.nan

        merged['amount_home'] = merged.apply(get_normalized_amount, axis=1)
        return merged

    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Retrieves a specific user's profile as a dictionary."""
        if self.profiles is None:
            return None
        user_row = self.profiles[self.profiles['user_id'] == user_id]
        if user_row.empty:
            return None
        return user_row.iloc[0].to_dict()

    def get_events_for_user(self, user_id: str, normalized_events: pd.DataFrame) -> pd.DataFrame:
        """Retrieves normalized events for a specific user."""
        return normalized_events[normalized_events['user_id'] == user_id]

    def get_payment_options(self, request_id: str) -> pd.DataFrame:
        """Retrieves payment options for a specific request."""
        if self.options is None:
            return pd.DataFrame()
        return self.options[self.options['request_id'] == request_id]
