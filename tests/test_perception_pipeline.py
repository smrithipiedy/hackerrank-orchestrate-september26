import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import date
from code.ingestion import DataIngestor
from code.perception_pipeline import PerceptionPipeline

class TestPerceptionPipeline(unittest.TestCase):

    def setUp(self):
        # Mock Ingestor
        self.ingestor = DataIngestor()
        self.ingestor.profiles = pd.DataFrame([
            {'user_id': 'user_1', 'home_currency': 'USD'}
        ])
        self.ingestor.events = pd.DataFrame([
            {'event_id': 'e1', 'user_id': 'user_1', 'amount': np.nan, 'currency': 'EUR', 'settlement_date': '2026-09-12', 'category': 'shopping', 'status': 'settled', 'direction': 'debit'}
        ])
        self.ingestor.images = pd.DataFrame([
            {'image_id': 'img_1', 'related_event_id': 'e1'}
        ])
        self.ingestor.rates = pd.DataFrame([
            {'rate_date': '2026-09-12', 'from_currency': 'EUR', 'to_currency': 'USD', 'rate': 1.1}
        ])
        self.ingestor.messages = pd.DataFrame(columns=['user_id', 'message_text', 'related_event_id'])

        # Mock the methods that would normally load from CSV
        self.ingestor.load_all = MagicMock()

        self.pipeline = PerceptionPipeline(self.ingestor, api_key=None)

    def test_image_to_normalized_amount_flow(self):
        """Verify the full pipeline from blank amount to normalized home currency value."""
        # Mock OCR to return a value
        self.pipeline.ocr.extract_amount = MagicMock(return_value=(100.0, "EUR"))

        # Mock normalize_events to return a basic dataframe for the user
        # (Since we are mocking the ingestor, we need to simulate what normalize_events does)
        def mock_normalize():
            df = self.ingestor.events.copy()
            df['amount_home'] = np.nan
            df['settlement_date'] = pd.to_datetime(df['settlement_date']).dt.date
            return df

        self.ingestor.normalize_events = MagicMock(side_effect=mock_normalize)
        self.ingestor.get_events_for_user = MagicMock(side_effect=lambda uid, df: df[df['user_id'] == uid])
        self.ingestor.get_user_profile = MagicMock(return_value={'home_currency': 'USD'})

        resolved_ledger = self.pipeline.process_all_users()

        # Check if e1 now has the correct normalized amount: 100 EUR * 1.1 = 110 USD
        event = resolved_ledger[resolved_ledger['event_id'] == 'e1'].iloc[0]
        self.assertEqual(event['amount'], 100.0)
        self.assertAlmostEqual(event['amount_home'], 110.0, places=5)

    def test_ocr_failure_fallback(self):
        """Verify that the pipeline handles OCR failure gracefully without inventing data."""
        # Mock OCR to return None
        self.pipeline.ocr.extract_amount = MagicMock(return_value=(None, None))

        def mock_normalize():
            df = self.ingestor.events.copy()
            df['amount_home'] = np.nan
            df['settlement_date'] = pd.to_datetime(df['settlement_date']).dt.date
            return df

        self.ingestor.normalize_events = MagicMock(side_effect=mock_normalize)
        self.ingestor.get_events_for_user = MagicMock(side_effect=lambda uid, df: df[df['user_id'] == uid])
        self.ingestor.get_user_profile = MagicMock(return_value={'home_currency': 'USD'})

        resolved_ledger = self.pipeline.process_all_users()

        event = resolved_ledger[resolved_ledger['event_id'] == 'e1'].iloc[0]
        self.assertTrue(pd.isna(event['amount_home']))

if __name__ == "__main__":
    unittest.main()
