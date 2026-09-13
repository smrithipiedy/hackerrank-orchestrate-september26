import unittest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from code.cashflow_simulator import CashFlowSimulator

class TestPhase3(unittest.TestCase):

    def setUp(self):
        self.profile = {
            'user_id': 'user_1',
            'current_available_balance': 1000.0,
            'minimum_balance_to_keep': 200.0,
            'home_currency': 'USD'
        }

    def test_basic_cash_flow(self):
        """Verify balance is correctly projected with a simple debit and credit."""
        ledger = pd.DataFrame([
            {'event_id': 'e1', 'user_id': 'user_1', 'amount_home': 100.0, 'direction': 'debit', 'settlement_date': date(2026, 9, 15), 'status': 'scheduled', 'category': 'util'},
            {'event_id': 'e2', 'user_id': 'user_1', 'amount_home': 500.0, 'direction': 'credit', 'settlement_date': date(2026, 9, 20), 'status': 'settled', 'category': 'salary'},
        ])
        sim = CashFlowSimulator(self.profile, ledger)
        req_date = date(2026, 9, 12)
        trace = sim.simulate(req_date)

        # Day 0 (Sept 12): 1000
        self.assertEqual(trace.daily_balances[0], 1000.0)
        # Day 3 (Sept 15): 1000 - 100 = 900
        self.assertEqual(trace.daily_balances[3], 900.0)
        # Day 8 (Sept 20): 900 + 500 = 1400
        self.assertEqual(trace.daily_balances[8], 1400.0)

    def test_min_balance_breach(self):
        """Verify that a breach is detected when balance falls below minimum."""
        ledger = pd.DataFrame([
            {'event_id': 'e1', 'user_id': 'user_1', 'amount_home': 900.0, 'direction': 'debit', 'settlement_date': date(2026, 9, 15), 'status': 'scheduled', 'category': 'rent'},
        ])
        sim = CashFlowSimulator(self.profile, ledger)
        req_date = date(2026, 9, 12)
        trace = sim.simulate(req_date)

        # Balance: 1000 -> 100. Cushion: 100 - 200 = -100.
        self.assertEqual(trace.min_cushion, -100.0)
        self.assertEqual(trace.first_breach_day, 3) # Sept 15
        self.assertIn('e1', trace.breach_cause_events)

    def test_pending_debit_reservation(self):
        """Verify pending debits are reserved while pending credits are ignored."""
        ledger = pd.DataFrame([
            {'event_id': 'd1', 'user_id': 'user_1', 'amount_home': 100.0, 'direction': 'debit', 'settlement_date': date(2026, 9, 13), 'status': 'pending', 'category': 'food'},
            {'event_id': 'c1', 'user_id': 'user_1', 'amount_home': 1000.0, 'direction': 'credit', 'settlement_date': date(2026, 9, 13), 'status': 'pending', 'category': 'bonus'},
        ])
        sim = CashFlowSimulator(self.profile, ledger)
        req_date = date(2026, 9, 12)
        trace = sim.simulate(req_date)

        # Day 1 (Sept 13): 1000 - 100 (pending debit) + 0 (pending credit ignored) = 900
        self.assertEqual(trace.daily_balances[1], 900.0)

    def test_ignored_events(self):
        """Verify cancelled, failed, and non-cash events are ignored."""
        ledger = pd.DataFrame([
            {'event_id': 'e1', 'user_id': 'user_1', 'amount_home': 100.0, 'direction': 'debit', 'settlement_date': date(2026, 9, 13), 'status': 'cancelled', 'category': 'food'},
            {'event_id': 'e2', 'user_id': 'user_1', 'amount_home': 200.0, 'direction': 'debit', 'settlement_date': date(2026, 9, 13), 'status': 'failed', 'category': 'food'},
            {'event_id': 'e3', 'user_id': 'user_1', 'amount_home': 300.0, 'direction': 'non_cash', 'settlement_date': date(2026, 9, 13), 'status': 'settled', 'category': 'investment'},
        ])
        sim = CashFlowSimulator(self.profile, ledger)
        req_date = date(2026, 9, 12)
        trace = sim.simulate(req_date)

        # Balance should remain 1000
        self.assertEqual(trace.daily_balances[1], 1000.0)

    def test_monthly_recurrence_projection(self):
        """Verify that monthly recurring income and expenses are projected forward."""
        ledger = pd.DataFrame([
            # Recurring Income: Salary on 1st of month
            {'event_id': 'i1', 'user_id': 'user_1', 'amount_home': 2000.0, 'direction': 'credit', 'settlement_date': date(2026, 7, 1), 'status': 'settled', 'category': 'salary'},
            {'event_id': 'i2', 'user_id': 'user_1', 'amount_home': 2000.0, 'direction': 'credit', 'settlement_date': date(2026, 8, 1), 'status': 'settled', 'category': 'salary'},
            {'event_id': 'i3', 'user_id': 'user_1', 'amount_home': 2000.0, 'direction': 'credit', 'settlement_date': date(2026, 9, 1), 'status': 'settled', 'category': 'salary'},

            # Recurring Expense: Rent on 15th of month
            {'event_id': 'e1', 'user_id': 'user_1', 'amount_home': 800.0, 'direction': 'debit', 'settlement_date': date(2026, 7, 15), 'status': 'settled', 'category': 'rent'},
            {'event_id': 'e2', 'user_id': 'user_1', 'amount_home': 800.0, 'direction': 'debit', 'settlement_date': date(2026, 8, 15), 'status': 'settled', 'category': 'rent'},
            {'event_id': 'e3', 'user_id': 'user_1', 'amount_home': 800.0, 'direction': 'debit', 'settlement_date': date(2026, 9, 15), 'status': 'settled', 'category': 'rent'},
        ])
        sim = CashFlowSimulator(self.profile, ledger)
        req_date = date(2026, 10, 1)
        trace = sim.simulate(req_date)

        # Oct 1: Income should hit (Day 0) -> 1000 + 2000 = 3000
        self.assertEqual(trace.daily_balances[0], 3000.0)
        # Oct 15: Rent should hit (Day 14) -> 3000 - 800 = 2200
        self.assertEqual(trace.daily_balances[14], 2200.0)
        # Nov 1: Income should hit (Day 31) -> 2200 + 2000 = 4200
        self.assertEqual(trace.daily_balances[31], 4200.0)
        # Nov 15: Rent should hit (Day 45) -> 4200 - 800 = 3400
        self.assertEqual(trace.daily_balances[45], 3400.0)

    def test_recurrence_month_end_drift(self):
        """Verify that recurrence anchored on 31st stays on the 31st (or end of month)."""
        ledger = pd.DataFrame([
            {'event_id': 'e1', 'user_id': 'user_1', 'amount_home': 100.0, 'direction': 'debit', 'settlement_date': date(2026, 1, 31), 'status': 'settled', 'category': 'sub'},
            {'event_id': 'e2', 'user_id': 'user_1', 'amount_home': 100.0, 'direction': 'debit', 'settlement_date': date(2026, 2, 28), 'status': 'settled', 'category': 'sub'},
            {'event_id': 'e3', 'user_id': 'user_1', 'amount_home': 100.0, 'direction': 'debit', 'settlement_date': date(2026, 3, 31), 'status': 'settled', 'category': 'sub'},
        ])
        sim = CashFlowSimulator(self.profile, ledger)
        req_date = date(2026, 4, 1)
        trace = sim.simulate(req_date)

        # April 30: Should be 1000 - 100 = 900
        self.assertEqual(trace.daily_balances[29], 900.0)
        # May 31: Should be 900 - 100 = 800
        self.assertEqual(trace.daily_balances[60], 800.0)

    def test_recurrence_double_counting(self):
        """Verify that projected recurrence does not double-count explicit future events."""
        ledger = pd.DataFrame([
            # History
            {'event_id': 'i1', 'user_id': 'user_1', 'amount_home': 2000.0, 'direction': 'credit', 'settlement_date': date(2026, 7, 1), 'status': 'settled', 'category': 'salary'},
            {'event_id': 'i2', 'user_id': 'user_1', 'amount_home': 2000.0, 'direction': 'credit', 'settlement_date': date(2026, 8, 1), 'status': 'settled', 'category': 'salary'},
            # Explicit future event on Oct 1
            {'event_id': 'i3', 'user_id': 'user_1', 'amount_home': 2000.0, 'direction': 'credit', 'settlement_date': date(2026, 10, 1), 'status': 'scheduled', 'category': 'salary'},
        ])
        sim = CashFlowSimulator(self.profile, ledger)
        req_date = date(2026, 10, 1)
        trace = sim.simulate(req_date)

        # Oct 1: Only one salary should hit -> 1000 + 2000 = 3000
        self.assertEqual(trace.daily_balances[0], 3000.0)

    def test_projected_breach_trace(self):
        """Verify projected recurring expenses appear in breach_cause_events."""
        # Min balance 200, balance 1000.
        # Recurring expense 900 on the 15th.
        ledger = pd.DataFrame([
            {'event_id': 'e1', 'user_id': 'user_1', 'amount_home': 900.0, 'direction': 'debit', 'settlement_date': date(2026, 7, 15), 'status': 'settled', 'category': 'rent'},
            {'event_id': 'e2', 'user_id': 'user_1', 'amount_home': 900.0, 'direction': 'debit', 'settlement_date': date(2026, 8, 15), 'status': 'settled', 'category': 'rent'},
        ])
        sim = CashFlowSimulator(self.profile, ledger)
        req_date = date(2026, 9, 1)
        trace = sim.simulate(req_date)

        # Sept 15: Balance 1000 - 900 = 100. Cushion = 100 - 200 = -100.
        self.assertTrue(trace.first_breach_day is not None)
        # The cause should be the projected rent event
        self.assertTrue(any("proj_debit_rent" in eid for eid in trace.breach_cause_events))

    def test_earliest_full_payment_date(self):
        """Verify finding the first date when a full payment is safe."""
        # User balance 500, min 200. Request 400.
        # Current cushion: 500 - 200 = 300. 300 < 400, not safe today.
        # Salary 1000 comes on Sept 20.
        profile = {'user_id': 'user_1', 'current_available_balance': 500.0, 'minimum_balance_to_keep': 200.0, 'home_currency': 'USD'}
        ledger = pd.DataFrame([
            {'event_id': 's1', 'user_id': 'user_1', 'amount_home': 1000.0, 'direction': 'credit', 'settlement_date': date(2026, 9, 20), 'status': 'scheduled', 'category': 'salary'},
        ])
        sim = CashFlowSimulator(profile, ledger)
        req_date = date(2026, 9, 12)

        earliest = sim.find_earliest_full_payment_date(400.0, req_date)
        # On Sept 20: Balance 500 + 1000 = 1500. 1500 - 400 = 1100. 1100 > 200. Safe.
        self.assertEqual(earliest, date(2026, 9, 20))

    def test_regression_dedup_and_breach_trace(self):
        """Verify date normalization, double-counting prevention, and credit-exclusion in breach trace."""
        # 1. Ledger with ISO string dates
        # User has a recurring expense: 100 on the 15th
        ledger = pd.DataFrame([
            {'event_id': 'e1', 'user_id': 'user_1', 'amount_home': 100.0, 'direction': 'debit', 'settlement_date': '2026-07-15', 'status': 'settled', 'category': 'sub'},
            {'event_id': 'e2', 'user_id': 'user_1', 'amount_home': 100.0, 'direction': 'debit', 'settlement_date': '2026-08-15', 'status': 'settled', 'category': 'sub'},
            # Explicit future event on Sept 15 - should NOT be double-counted by projection
            {'event_id': 'e3', 'user_id': 'user_1', 'amount_home': 100.0, 'direction': 'debit', 'settlement_date': '2026-09-15', 'status': 'scheduled', 'category': 'sub'},
            # Recurring income: 200 on the 15th (to test projected credit in breach)
            {'event_id': 'i1', 'user_id': 'user_1', 'amount_home': 200.0, 'direction': 'credit', 'settlement_date': '2026-07-15', 'status': 'settled', 'category': 'bonus'},
            {'event_id': 'i2', 'user_id': 'user_1', 'amount_home': 200.0, 'direction': 'credit', 'settlement_date': '2026-08-15', 'status': 'settled', 'category': 'bonus'},
        ])

        # Adjust profile to force a breach on Sept 15
        # Balance 300, Min 200.
        self.profile['current_available_balance'] = 250.0
        # Make the debit 500.
        ledger.loc[ledger['event_id'].isin(['e1', 'e2', 'e3']), 'amount_home'] = 500.0

        # Sept 15: 250 - 500 (e3) + 200 (proj_credit) = -50. Cushion = -250. BREACH!
        sim = CashFlowSimulator(self.profile, ledger)
        req_date = date(2026, 9, 1)
        trace = sim.simulate(req_date)

        # Check double-counting: only e3 should be present, not proj_sub_500_...
        # Balance should be exactly -50.
        self.assertEqual(trace.daily_balances[14], -50.0)

        # Check breach cause: projected credit should NOT be in causes
        self.assertIn('e3', trace.breach_cause_events)
        # Verify no projected credit is listed
        for cause in trace.breach_cause_events:
            self.assertFalse(cause.startswith('proj_credit_'))

if __name__ == "__main__":
    unittest.main()
