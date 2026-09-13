import unittest
import pandas as pd
from datetime import date
from code.schemas import OutputRow
from code.explanation_generator import ExplanationGenerator
from code.decision_engine import DecisionEngine

class TestPhase6(unittest.TestCase):

    def setUp(self):
        self.explainer = ExplanationGenerator()
        self.engine = DecisionEngine()
        self.profile = {
            'user_id': 'user_1',
            'current_available_balance': 1000.0,
            'minimum_balance_to_keep': 200.0,
            'home_currency': 'USD',
            'payment_methods_user_will_consider': 'full_payment|partial_payment|installments'
        }

    def test_schema_valid_row(self):
        """Test OutputRow valid construction."""
        row_dict = {
            'request_id': 'req_01',
            'amount_safe_to_pay': 500.0,
            'affordability_status': 'affordable_now',
            'recommended_payment_method': 'full_payment',
            'payment_plan': '2026-09-01:500',
            'earliest_date_for_full_payment': '2026-09-01',
            'spending_changes_needed': 'none',
            'decision_explanation': 'Pay USD 500 today. This leaves at least USD 200 available over the next 90 days.'
        }
        row = OutputRow.model_validate(row_dict)
        self.assertEqual(row.request_id, 'req_01')
        self.assertEqual(row.affordability_status, 'affordable_now')

    def test_schema_invalid_status(self):
        """Test OutputRow rejects invalid affordability_status enum."""
        row_dict = {
            'request_id': 'req_01',
            'amount_safe_to_pay': 500.0,
            'affordability_status': 'unknown_status',
            'recommended_payment_method': 'full_payment',
            'payment_plan': '2026-09-01:500',
            'earliest_date_for_full_payment': '2026-09-01',
            'spending_changes_needed': 'none',
            'decision_explanation': 'test'
        }
        with self.assertRaises(Exception):
            OutputRow.model_validate(row_dict)

    def test_schema_invalid_method(self):
        """Test OutputRow rejects invalid recommended_payment_method enum."""
        row_dict = {
            'request_id': 'req_01',
            'amount_safe_to_pay': 500.0,
            'affordability_status': 'affordable_now',
            'recommended_payment_method': 'crypto_transfer',
            'payment_plan': '2026-09-01:500',
            'earliest_date_for_full_payment': '2026-09-01',
            'spending_changes_needed': 'none',
            'decision_explanation': 'test'
        }
        with self.assertRaises(Exception):
            OutputRow.model_validate(row_dict)

    def test_schema_invalid_plan_format(self):
        """Test OutputRow rejects invalid payment_plan formatting."""
        row_dict = {
            'request_id': 'req_01',
            'amount_safe_to_pay': 500.0,
            'affordability_status': 'affordable_now',
            'recommended_payment_method': 'full_payment',
            'payment_plan': 'invalid_date:500',
            'earliest_date_for_full_payment': '2026-09-01',
            'spending_changes_needed': 'none',
            'decision_explanation': 'test'
        }
        with self.assertRaises(Exception):
            OutputRow.model_validate(row_dict)

    def test_schema_invalid_spending_changes(self):
        """Test OutputRow rejects more than 3 spending changes."""
        row_dict = {
            'request_id': 'req_01',
            'amount_safe_to_pay': 500.0,
            'affordability_status': 'affordable_with_plan',
            'recommended_payment_method': 'full_payment',
            'payment_plan': '2026-09-01:500',
            'earliest_date_for_full_payment': '2026-09-01',
            'spending_changes_needed': 'stop:e1|stop:e2|stop:e3|stop:e4',
            'decision_explanation': 'test'
        }
        with self.assertRaises(Exception):
            OutputRow.model_validate(row_dict)

    def test_explanation_generation_affordable_now(self):
        """Test grounded explanation for affordable_now full payment."""
        request = {
            'request_id': 'req_01',
            'requested_amount': 500.0,
            'request_date': date(2026, 9, 1),
            'desired_completion_date': date(2026, 9, 20)
        }
        decision = {
            'recommended_payment_method': 'full_payment',
            'affordability_status': 'affordable_now',
            'spending_changes_needed': 'none'
        }
        best_plan = {
            'method': 'full_payment',
            'payments': [(date(2026, 9, 1), 500.0)]
        }
        exp = self.explainer.generate_explanation(request, self.profile, decision, best_plan)
        self.assertIn("Pay USD 500 today", exp)
        self.assertIn("USD 200 available", exp)

    def test_explanation_generation_not_affordable(self):
        """Test grounded explanation for not_affordable fallback."""
        request = {
            'request_id': 'req_02',
            'requested_amount': 5000.0,
            'request_date': date(2026, 9, 1),
            'desired_completion_date': date(2026, 9, 20)
        }
        decision = {
            'recommended_payment_method': 'not_recommended',
            'affordability_status': 'not_affordable',
            'spending_changes_needed': 'none',
            'amount_safe_to_pay': 0.0
        }
        exp = self.explainer.generate_explanation(request, self.profile, decision, None)
        self.assertIn("Do not make this payment", exp)
        self.assertIn("20 September 2026", exp)
        self.assertIn("USD 200 minimum protected", exp)

if __name__ == "__main__":
    unittest.main()
