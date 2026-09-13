import unittest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from code.cashflow_simulator import CashFlowSimulator
from code.plan_optimizer import PlanOptimizer

class TestPhase4(unittest.TestCase):

    def setUp(self):
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
        """Helper to add an event to the ledger."""
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

    def test_affordable_now(self):
        """Full payment is safe today."""
        self._add_event('e1', 100.0, 'debit', 'util', days_offset=3)  # Sept 4
        request = {
            'request_id': 'req_1',
            'request_date': date(2026, 9, 1),
            'requested_amount': 500.0,
            'desired_completion_date': date(2026, 9, 20),
            'allows_partial_payment': True
        }
        options = pd.DataFrame(columns=['request_id'])
        res = self.optimizer.optimize('req_1', request, self.profile, self.ledger, options)
        self.assertEqual(res['recommended_payment_method'], 'full_payment')
        self.assertEqual(res['affordability_status'], 'affordable_now')
        self.assertEqual(res['amount_safe_to_pay'], 500.0) # Capped at requested_amount
        self.assertEqual(res['earliest_date_for_full_payment'], '2026-09-01')

    def test_installment_required(self):
        """Full payment not safe today due to future expense, but installments safe with interim income."""
        # Ledger: expense of 500 on Sept 4, income of 300 on Sept 15
        self._add_event('e1', 500.0, 'debit', 'rent', days_offset=3)   # Sept 4
        self._add_event('i1', 300.0, 'credit', 'salary', days_offset=14) # Sept 15
        # Balance 1000, min 200.
        # Request 500 on Sept 1.
        # Full payment on Sept 1:
        #   Day0: 1000-500=500
        #   Day3: 500-500=0 -> below min 200 -> not safe.
        request = {
            'request_id': 'req_2',
            'request_date': date(2026, 9, 1),
            'requested_amount': 500.0,
            'desired_completion_date': date(2026, 10, 5),
            'allows_partial_payment': False
        }
        # Installment: 2 payments of 250 on Sept 1 and Oct 1.
        options = pd.DataFrame([{
            'request_id': 'req_2',
            'payment_option_id': 'payment_option_01',
            'payment_method': 'installments',
            'payment_amount': 250.0,
            'first_payment_date': date(2026, 9, 1),
            'number_of_payments': 2,
            'payment_frequency_days': 30,
            'total_payable_amount': 500.0
        }])
        res = self.optimizer.optimize('req_2', request, self.profile, self.ledger, options)
        self.assertEqual(res['recommended_payment_method'], 'installments')
        self.assertEqual(res['affordability_status'], 'affordable_with_plan')
        self.assertEqual(res['payment_plan'], '2026-09-01:250|2026-10-01:250')

    def test_partial_payment_constraints(self):
        """Verify exact partial payment rules."""
        self._add_event('e1', 600.0, 'debit', 'rent', days_offset=3)  # Sept 4
        self._add_event('s1', 1000.0, 'credit', 'salary', days_offset=17)  # Sept 18
        # Balance 1000, min 200. Safe today: 1000 - 600 - 200 = 200.
        request = {
            'request_id': 'req_3',
            'request_date': date(2026, 9, 1),
            'requested_amount': 500.0,
            'desired_completion_date': date(2026, 9, 20),
            'allows_partial_payment': True
        }
        options = pd.DataFrame(columns=['request_id', 'payment_method'])
        res = self.optimizer.optimize('req_3', request, self.profile, self.ledger, options)
        self.assertEqual(res['recommended_payment_method'], 'partial_payment')
        self.assertEqual(res['affordability_status'], 'affordable_with_plan')
        self.assertEqual(res['payment_plan'], '2026-09-01:200|2026-09-18:300')
        self.assertEqual(res['amount_safe_to_pay'], 200.0)
        self.assertEqual(res['earliest_date_for_full_payment'], '2026-09-18')

    def test_affordable_later_wait(self):
        """Wait is selected when full payment becomes safe later on or before deadline."""
        self._add_event('e1', 700.0, 'debit', 'rent', days_offset=3)  # Sept 4
        self._add_event('s1', 1000.0, 'credit', 'salary', days_offset=19)  # Sept 20
        # Balance 1000, min 200. Safe today: 100.
        request = {
            'request_id': 'req_4',
            'request_date': date(2026, 9, 1),
            'requested_amount': 500.0,
            'desired_completion_date': date(2026, 10, 1),
            'allows_partial_payment': False
        }
        options = pd.DataFrame(columns=['request_id', 'payment_method'])
        res = self.optimizer.optimize('req_4', request, self.profile, self.ledger, options)
        self.assertEqual(res['recommended_payment_method'], 'wait')
        self.assertEqual(res['affordability_status'], 'affordable_later')
        self.assertEqual(res['earliest_date_for_full_payment'], '2026-09-20')
        self.assertEqual(res['payment_plan'], '2026-09-20:500')

    def test_spending_changes_exhaustive(self):
        """Verify that flexible spending reduction renders a plan safe."""
        # Balance 1000, min 200. Request 700.
        # Without changes: e1 (dining, debit 200) occurs on Sept 4.
        # Balance on Sept 4 before request: 1000 - 200 = 800.
        # Cushion with 700 payment on Sept 1: 1000 - 700 - 200(e1) = 100 < 200 (min balance)!
        # So not safe without spending change.
        # If e1 is reduced to 50:
        # Cushion with 700 payment: 1000 - 700 - 50 = 250 >= 200 (min balance)! SAFE!
        self._add_event('e1', 200.0, 'debit', 'dining', days_offset=3, min_allowed=50.0, flex='reducible')
        request = {
            'request_id': 'req_5',
            'request_date': date(2026, 9, 1),
            'requested_amount': 700.0,
            'desired_completion_date': date(2026, 9, 20),
            'allows_partial_payment': False
        }
        options = pd.DataFrame(columns=['request_id', 'payment_method'])
        res = self.optimizer.optimize('req_5', request, self.profile, self.ledger, options)
        self.assertEqual(res['recommended_payment_method'], 'full_payment')
        self.assertEqual(res['affordability_status'], 'affordable_with_plan')
        self.assertEqual(res['spending_changes_needed'], 'reduce_to:e1:50')

    def test_tie_breaking(self):
        """Verify the 6-tier ranking: Full payment preferred over Installments when both safe."""
        self._add_event('e1', 100.0, 'debit', 'util', days_offset=3)  # Sept 4
        request = {
            'request_id': 'req_6',
            'request_date': date(2026, 9, 1),
            'requested_amount': 500.0,
            'desired_completion_date': date(2026, 9, 20),
            'allows_partial_payment': True
        }
        options = pd.DataFrame([{
            'request_id': 'req_6',
            'payment_option_id': 'payment_option_01',
            'payment_method': 'installments',
            'payment_amount': 250.0,
            'first_payment_date': date(2026, 9, 1),
            'number_of_payments': 2,
            'payment_frequency_days': 30,
            'total_payable_amount': 500.0
        }])
        res = self.optimizer.optimize('req_6', request, self.profile, self.ledger, options)
        # Full payment uses fewer payments (Tier 5: 1 vs 2), so full payment wins
        self.assertEqual(res['recommended_payment_method'], 'full_payment')
        self.assertEqual(res['affordability_status'], 'affordable_now')

    def test_not_affordable(self):
        """No safe plan available even with spending changes."""
        self._add_event('e1', 900.0, 'debit', 'rent', days_offset=3, flex='fixed')
        request = {
            'request_id': 'req_7',
            'request_date': date(2026, 9, 1),
            'requested_amount': 1000.0,
            'desired_completion_date': date(2026, 9, 20),
            'allows_partial_payment': False
        }
        options = pd.DataFrame(columns=['request_id', 'payment_method'])
        res = self.optimizer.optimize('req_7', request, self.profile, self.ledger, options)
        self.assertEqual(res['recommended_payment_method'], 'not_recommended')
        self.assertEqual(res['affordability_status'], 'not_affordable')
        self.assertEqual(res['payment_plan'], 'none')
        self.assertEqual(res['spending_changes_needed'], 'none')

    def test_max_installment_months_filter(self):
        """Options exceeding user's max_installment_months are rejected."""
        self.profile['max_installment_months'] = 3
        # Request 1000 on Sept 1, user considers installments
        self._add_event('e1', 700.0, 'debit', 'rent', days_offset=3)
        request = {
            'request_id': 'req_8',
            'request_date': date(2026, 9, 1),
            'requested_amount': 600.0,
            'desired_completion_date': date(2027, 9, 1),
            'allows_partial_payment': False
        }
        # Option with 12 months duration (span: 11 * 30 days = 330 days / 30 = 11 months > 3)
        options = pd.DataFrame([{
            'request_id': 'req_8',
            'payment_option_id': 'opt_long',
            'payment_method': 'installments',
            'payment_amount': 50.0,
            'first_payment_date': date(2026, 9, 1),
            'number_of_payments': 12,
            'payment_frequency_days': 30,
            'total_payable_amount': 600.0
        }])
        res = self.optimizer.optimize('req_8', request, self.profile, self.ledger, options)
        # Should be rejected because it exceeds max_installment_months of 3
        self.assertNotEqual(res['recommended_payment_method'], 'installments')

    def test_payment_option_id_safe_ranking(self):
        """Tie breaking on payment_option_id compares string IDs safely without assuming int parsing."""
        self._add_event('e1', 500.0, 'debit', 'rent', days_offset=3)
        self._add_event('s1', 500.0, 'credit', 'salary', days_offset=15)
        # Only installments considered
        self.profile['payment_methods_user_will_consider'] = 'installments'
        request = {
            'request_id': 'req_9',
            'request_date': date(2026, 9, 1),
            'requested_amount': 400.0,
            'desired_completion_date': date(2026, 11, 1),
            'allows_partial_payment': False
        }
        # Two identical installment options with arbitrary string IDs: "opt_B" and "opt_A"
        options = pd.DataFrame([
            {
                'request_id': 'req_9',
                'payment_option_id': 'opt_B',
                'payment_method': 'installments',
                'payment_amount': 200.0,
                'first_payment_date': date(2026, 9, 1),
                'number_of_payments': 2,
                'payment_frequency_days': 30,
                'total_payable_amount': 400.0
            },
            {
                'request_id': 'req_9',
                'payment_option_id': 'opt_A',
                'payment_method': 'installments',
                'payment_amount': 200.0,
                'first_payment_date': date(2026, 9, 1),
                'number_of_payments': 2,
                'payment_frequency_days': 30,
                'total_payable_amount': 400.0
            }
        ])
        res = self.optimizer.optimize('req_9', request, self.profile, self.ledger, options)
        self.assertEqual(res['recommended_payment_method'], 'installments')
        # Lexicographically 'opt_A' < 'opt_B'
        self.assertEqual(res['payment_plan'], '2026-09-01:200|2026-10-01:200')

if __name__ == "__main__":
    unittest.main()
