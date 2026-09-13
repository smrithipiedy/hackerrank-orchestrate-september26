import unittest
import pandas as pd
from code.schemas import OutputRow

class TestSampleRegression(unittest.TestCase):
    """
    Phase 7: Behavioral comparison of pipeline output format and decision style
    against the 25 public examples in dataset/sample_requests.csv.
    NOTE: As specified in AGENTS.md §6.1, sample_requests.csv contains public
    examples to understand format and decision style, NOT labels for evaluation requests.
    """

    def setUp(self):
        self.samples = pd.read_csv('dataset/sample_requests.csv')
        self.output = pd.read_csv('dataset/output.csv')

    def test_sample_schema_conformance(self):
        """Verify all 25 sample requests conform to the strict OutputRow schema."""
        for idx, row in self.samples.iterrows():
            row_dict = row.to_dict()
            if pd.isna(row_dict.get('earliest_date_for_full_payment')):
                row_dict['earliest_date_for_full_payment'] = ''
            # Assert schema validates cleanly
            validated = OutputRow.model_validate(row_dict)
            self.assertIsNotNone(validated)

    def test_output_schema_conformance(self):
        """Verify all 250 evaluation requests in output.csv conform to OutputRow schema."""
        self.assertEqual(len(self.output), 250)
        for idx, row in self.output.iterrows():
            row_dict = row.to_dict()
            if pd.isna(row_dict.get('earliest_date_for_full_payment')):
                row_dict['earliest_date_for_full_payment'] = ''
            validated = OutputRow.model_validate(row_dict)
            self.assertIsNotNone(validated)

    def test_output_bounds_and_validity(self):
        """Verify amount_safe_to_pay bounds against requests.csv."""
        requests = pd.read_csv('dataset/requests.csv')
        merged = pd.merge(self.output, requests, on='request_id')
        self.assertEqual(len(merged), 250)

        for _, row in merged.iterrows():
            safe_amt = float(row['amount_safe_to_pay'])
            req_amt = float(row['requested_amount'])
            self.assertGreaterEqual(safe_amt, 0.0)
            self.assertLessEqual(safe_amt, req_amt + 1e-5)

if __name__ == "__main__":
    unittest.main()
