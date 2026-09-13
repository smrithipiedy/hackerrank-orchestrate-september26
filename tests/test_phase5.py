import unittest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from code.decision_engine import DecisionEngine
from code.plan_optimizer import PlanOptimizer

class TestPhase5(unittest.TestCase):

    def setUp(self):
        self.engine = DecisionEngine()
        self.optimizer = PlanOptimizer()
        self.profile = {
            'user_id': 'user_1',
            'current_available_balance': 1000.0,
            'minimum_balance_to_keep': 200.0,
            'home_currency': 'USD',
            'payment_methods_user_will_consider': 'full_payment|partial_payment|installments',
            'expense_categories_user_is_willing_to_stop': 'streaming',
            'expense_categories_user_is_willing_to_reduce': 'dining',
            'max_installment_months': 12
        }
        self.events_list = []

    def _add_event(self, eid, amt, direction, cat, status='settled', days_offset=0, min_allowed=None, flex='fixed'):
        event_date = date(2026, 9, 1) + timedelta(days=days_offset)
        self.events_list.append({
            'event_id': eid,
            'user_id': self.profile['user_id'],
            'event_type': 'expense' if direction == 'debit' else 'income',
            'description': cat,
            'category': cat,
            'direction': direction,
            'amount': amt,
            'currency': self.profile['home_currency'],
            'event_date': event_date,
            'settlement_date': event_date,
            'status': status,
            'linked_event_id': '',
            'flexibility': flex,
            'minimum_allowed_amount': min_allowed if min_allowed is not None else np.nan,
            'amount_home': amt
        })
        self.ledger = pd.DataFrame(self.events_list)

    def test_tier1_deadline_compliance(self):
        """Tier 1: Plans completing by desired_completion_date beat plans completing after deadline."""
        deadline = date(2026, 9, 15)
        # Plan A: completes on Sept 10 (<= deadline), costs $500
        plan_a = {
            'method': 'full_payment',
            'payments': [(date(2026, 9, 10), 500.0)],
            'total_amount': 500.0,
            'completion_date': date(2026, 9, 10),
            'first_payment_date': date(2026, 9, 10),
            'num_payments': 1,
            'spending_changes': [],
            'payment_option_id': None
        }
        # Plan B: completes on Sept 20 (> deadline), costs $400 (cheaper, but misses deadline)
        plan_b = {
            'method': 'wait',
            'payments': [(date(2026, 9, 20), 400.0)],
            'total_amount': 400.0,
            'completion_date': date(2026, 9, 20),
            'first_payment_date': date(2026, 9, 20),
            'num_payments': 1,
            'spending_changes': [],
            'payment_option_id': None
        }
        best = self.engine.select_best_plan([plan_b, plan_a], deadline)
        self.assertEqual(best, plan_a)

    def test_tier2_spending_changes(self):
        """Tier 2: Plans requiring no spending changes beat plans requiring changes."""
        deadline = date(2026, 9, 20)
        # Plan A: requires spending changes, total $450
        plan_a = {
            'method': 'full_payment',
            'payments': [(date(2026, 9, 1), 450.0)],
            'total_amount': 450.0,
            'completion_date': date(2026, 9, 1),
            'first_payment_date': date(2026, 9, 1),
            'num_payments': 1,
            'spending_changes': ['stop:e1'],
            'payment_option_id': None
        }
        # Plan B: requires NO spending changes, total $500
        plan_b = {
            'method': 'installments',
            'payments': [(date(2026, 9, 1), 250.0), (date(2026, 9, 15), 250.0)],
            'total_amount': 500.0,
            'completion_date': date(2026, 9, 15),
            'first_payment_date': date(2026, 9, 1),
            'num_payments': 2,
            'spending_changes': [],
            'payment_option_id': 'opt_1'
        }
        best = self.engine.select_best_plan([plan_a, plan_b], deadline)
        self.assertEqual(best, plan_b)

    def test_tier3_total_cost(self):
        """Tier 3: Minimize total amount paid."""
        deadline = date(2026, 9, 20)
        # Plan A: total $520
        plan_a = {
            'method': 'installments',
            'payments': [(date(2026, 9, 1), 260.0), (date(2026, 9, 15), 260.0)],
            'total_amount': 520.0,
            'completion_date': date(2026, 9, 15),
            'first_payment_date': date(2026, 9, 1),
            'num_payments': 2,
            'spending_changes': [],
            'payment_option_id': 'opt_1'
        }
        # Plan B: total $500
        plan_b = {
            'method': 'installments',
            'payments': [(date(2026, 9, 1), 250.0), (date(2026, 9, 15), 250.0)],
            'total_amount': 500.0,
            'completion_date': date(2026, 9, 15),
            'first_payment_date': date(2026, 9, 1),
            'num_payments': 2,
            'spending_changes': [],
            'payment_option_id': 'opt_2'
        }
        best = self.engine.select_best_plan([plan_a, plan_b], deadline)
        self.assertEqual(best, plan_b)

    def test_tier4_earlier_start(self):
        """Tier 4: Start payment earlier."""
        deadline = date(2026, 9, 20)
        # Plan A: starts Sept 5
        plan_a = {
            'method': 'installments',
            'payments': [(date(2026, 9, 5), 250.0), (date(2026, 9, 15), 250.0)],
            'total_amount': 500.0,
            'completion_date': date(2026, 9, 15),
            'first_payment_date': date(2026, 9, 5),
            'num_payments': 2,
            'spending_changes': [],
            'payment_option_id': 'opt_1'
        }
        # Plan B: starts Sept 1
        plan_b = {
            'method': 'installments',
            'payments': [(date(2026, 9, 1), 250.0), (date(2026, 9, 15), 250.0)],
            'total_amount': 500.0,
            'completion_date': date(2026, 9, 15),
            'first_payment_date': date(2026, 9, 1),
            'num_payments': 2,
            'spending_changes': [],
            'payment_option_id': 'opt_2'
        }
        best = self.engine.select_best_plan([plan_a, plan_b], deadline)
        self.assertEqual(best, plan_b)

    def test_tier5_fewer_payments(self):
        """Tier 5: Use fewer payments."""
        deadline = date(2026, 9, 20)
        # Plan A: 3 payments
        plan_a = {
            'method': 'installments',
            'payments': [(date(2026, 9, 1), 100.0), (date(2026, 9, 5), 100.0), (date(2026, 9, 10), 100.0)],
            'total_amount': 300.0,
            'completion_date': date(2026, 9, 10),
            'first_payment_date': date(2026, 9, 1),
            'num_payments': 3,
            'spending_changes': [],
            'payment_option_id': 'opt_1'
        }
        # Plan B: 2 payments
        plan_b = {
            'method': 'installments',
            'payments': [(date(2026, 9, 1), 150.0), (date(2026, 9, 10), 150.0)],
            'total_amount': 300.0,
            'completion_date': date(2026, 9, 10),
            'first_payment_date': date(2026, 9, 1),
            'num_payments': 2,
            'spending_changes': [],
            'payment_option_id': 'opt_2'
        }
        best = self.engine.select_best_plan([plan_a, plan_b], deadline)
        self.assertEqual(best, plan_b)

    def test_tier6_safe_option_id(self):
        """Tier 6: Use lowest payment_option_id as string comparison."""
        deadline = date(2026, 9, 20)
        plan_a = {
            'method': 'installments',
            'payments': [(date(2026, 9, 1), 150.0), (date(2026, 9, 10), 150.0)],
            'total_amount': 300.0,
            'completion_date': date(2026, 9, 10),
            'first_payment_date': date(2026, 9, 1),
            'num_payments': 2,
            'spending_changes': [],
            'payment_option_id': 'payment_option_B'
        }
        plan_b = {
            'method': 'installments',
            'payments': [(date(2026, 9, 1), 150.0), (date(2026, 9, 10), 150.0)],
            'total_amount': 300.0,
            'completion_date': date(2026, 9, 10),
            'first_payment_date': date(2026, 9, 1),
            'num_payments': 2,
            'spending_changes': [],
            'payment_option_id': 'payment_option_A'
        }
        best = self.engine.select_best_plan([plan_a, plan_b], deadline)
        self.assertEqual(best, plan_b)

    def test_fallback_not_affordable(self):
        """Fallback when no safe candidates exist."""
        res = self.engine.format_decision(
            request_id='req_empty',
            best_plan=None,
            req_date=date(2026, 9, 1),
            amount_safe_to_pay=50.0,
            earliest_date_for_full_payment=None
        )
        self.assertEqual(res['recommended_payment_method'], 'not_recommended')
        self.assertEqual(res['affordability_status'], 'not_affordable')
        self.assertEqual(res['payment_plan'], 'none')
        self.assertEqual(res['spending_changes_needed'], 'none')
        self.assertEqual(res['earliest_date_for_full_payment'], '')

    def test_evaluate_request_integration(self):
        """End-to-end evaluation with DecisionEngine wrapping PlanOptimizer."""
        self._add_event('e1', 100.0, 'debit', 'util', days_offset=3)
        request = {
            'request_id': 'req_eval',
            'request_date': date(2026, 9, 1),
            'requested_amount': 500.0,
            'desired_completion_date': date(2026, 9, 20),
            'allows_partial_payment': True
        }
        options = pd.DataFrame(columns=['request_id'])
        res = self.engine.evaluate_request('req_eval', request, self.profile, self.ledger, options)
        self.assertEqual(res['recommended_payment_method'], 'full_payment')
        self.assertEqual(res['affordability_status'], 'affordable_now')
        self.assertEqual(res['amount_safe_to_pay'], 500.0)
        self.assertEqual(res['payment_plan'], '2026-09-01:500')

if __name__ == "__main__":
    unittest.main()
