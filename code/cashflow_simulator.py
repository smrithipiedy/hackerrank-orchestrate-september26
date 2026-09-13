import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from datetime import date, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SimulationAuditTrace:
    """Deterministic audit trace of the 90-day balance projection."""
    daily_balances: List[float]
    daily_cushions: List[float]
    min_cushion: float
    first_breach_day: Optional[int]
    breach_cause_events: List[str]

class CashFlowSimulator:
    """
    Deterministic 90-Day Cash Flow Simulation Engine.
    Projects daily balances to ensure minimum balance compliance.
    """

    def __init__(self, user_profile: Dict[str, Any], cleaned_ledger: pd.DataFrame):
        """
        Args:
            user_profile: User profile including 'current_available_balance' and 'minimum_balance_to_keep'.
            cleaned_ledger: The resolved, normalized ledger from Phase 2.
        """
        self.user_id = user_profile.get('user_id')
        self.start_balance = float(user_profile.get('current_available_balance', 0.0))
        self.min_balance = float(user_profile.get('minimum_balance_to_keep', 0.0))
        self.ledger = cleaned_ledger
        self.home_currency = user_profile.get('home_currency')

    def simulate(self, request_date: date, additional_payments: List[Tuple[date, float]] = None) -> SimulationAuditTrace:
        """
        Performs a day-by-day simulation for 90 days following the request_date.

        Args:
            request_date: The date the simulation begins (t=0).
            additional_payments: List of (date, amount) to subtract during simulation.

        Returns:
            SimulationAuditTrace containing the full mathematical history.
        """
        projection_days = 90
        daily_deltas = {}
        day_to_event_ids = {}

        def add_delta(d: date, amount: float, eid: Optional[str] = None):
            daily_deltas[d] = daily_deltas.get(d, 0.0) + amount
            if eid:
                day_to_event_ids.setdefault(d, []).append(eid)

        # 1. Process Ledger Events (Confirmed/Scheduled)
        # Get all user events first
        user_ledger = self.ledger[self.ledger['user_id'] == self.user_id].copy()

        # Filter for events on or after request_date
        # and exclude cancelled, unrealized, failed
        relevant_events = user_ledger[
            (pd.to_datetime(user_ledger['settlement_date']).dt.date >= request_date) &
            (~user_ledger['status'].isin(['cancelled', 'unrealized', 'failed']))
        ].copy()

        # Normalize settlement_date to date objects for deterministic comparison
        relevant_events['settlement_date'] = pd.to_datetime(relevant_events['settlement_date']).dt.date


        for _, event in relevant_events.iterrows():
            event_date = pd.to_datetime(event['settlement_date']).date()
            amount = float(event['amount_home'])
            eid = event['event_id']

            # Rule: Reserve pending debits. Ignore pending credits until they settle.
            status = event['status']
            direction = event['direction']

            # STRICT RULE: non_cash events must be ignored entirely.
            if direction == 'non_cash':
                continue

            if direction == 'debit':
                # All debits (settled, scheduled, pending) are subtracted
                add_delta(event_date, -amount, eid)
            elif direction == 'credit':
                # Only settled or scheduled credits are added
                if status in ['settled', 'scheduled']:
                    add_delta(event_date, amount, eid)

        # 2. Recurrence Projection
        # Detect recurring patterns from historical data and project them forward
        self._project_recurrence(request_date, user_ledger, relevant_events, add_delta)

        # 3. Apply Additional Payments (the request itself)
        if additional_payments:
            for p_date, p_amount in additional_payments:
                # Use a synthetic ID for additional payments
                add_delta(p_date, -p_amount, f"payment_{p_date}")

        # 4. Day-by-Day Calculation
        balances = []
        cushions = []
        current_balance = self.start_balance
        first_breach_day = None
        breach_cause_events = []

        for t in range(projection_days):
            curr_date = request_date + timedelta(days=t)
            current_balance += daily_deltas.get(curr_date, 0.0)

            balances.append(current_balance)
            cushion = current_balance - self.min_balance
            cushions.append(cushion)

            if cushion < 0 and first_breach_day is None:
                first_breach_day = t
                # Find events on this day contributing to the breach
                # We must identify which events on this day were debits.
                # We can use the ledger for explicit events and synthetic IDs for projected ones.
                day_events = day_to_event_ids.get(curr_date, [])

                causes = []
                for eid in day_events:
                    # Check if it's a projected DEBIT event
                    if eid.startswith('proj_debit_'):
                        causes.append(eid)
                    else:
                        # Check explicit event in ledger
                        event = user_ledger[user_ledger['event_id'] == eid]
                        if not event.empty and event.iloc[0]['direction'] == 'debit':
                            causes.append(eid)

                breach_cause_events.extend(causes)

        return SimulationAuditTrace(
            daily_balances=balances,
            daily_cushions=cushions,
            min_cushion=min(cushions) if cushions else 0.0,
            first_breach_day=first_breach_day,
            breach_cause_events=breach_cause_events
        )

    def _project_recurrence(self, request_date: date, user_ledger: pd.DataFrame, relevant_events: pd.DataFrame, add_delta_fn):
        """
        Detects recurring patterns in historical data and projects them into the 90-day window.
        A recurring event is defined as the same (category, amount_home, direction) occurring
        at least twice with a consistent monthly interval.
        """
        # Filter for historical events before request_date
        historical = user_ledger[pd.to_datetime(user_ledger['settlement_date']).dt.date < request_date].copy()
        if historical.empty:
            return

        # Group by identifying characteristics
        groups = historical.groupby(['category', 'amount_home', 'direction'])

        for (cat, amt, direct), group in groups:
            if len(group) < 2:
                continue

            # Sort by date to check intervals
            sorted_dates = pd.to_datetime(group['settlement_date']).dt.date.sort_values()

            # Check for monthly recurrence: same day of month for most occurrences
            is_monthly = True
            last_date = sorted_dates.iloc[-1]

            # Verify consistent monthly pattern (day of month should be similar)
            days_of_month = [d.day for d in sorted_dates]
            if len(days_of_month) > 0:
                most_common_day = max(set(days_of_month), key=days_of_month.count)
                if days_of_month.count(most_common_day) < 2:
                    is_monthly = False

            if not is_monthly:
                continue

            target_day = most_common_day

            # Project monthly occurrences forward
            # We project from the original base month/year of the first occurrence
            # but effectively we just need the next occurrence after last_date.
            curr_month = last_date.month
            curr_year = last_date.year

            while True:
                # Advance to next month
                curr_month += 1
                if curr_month > 12:
                    curr_month = 1
                    curr_year += 1

                # Calculate the date for this month using the original target_day
                import calendar
                days_in_month = calendar.monthrange(curr_year, curr_month)[1]
                proj_day = min(target_day, days_in_month)
                proj_date = date(curr_year, curr_month, proj_day)

                if proj_date > request_date + timedelta(days=90):
                    break

                if proj_date >= request_date:
                    # DOUBLE-COUNTING CHECK:
                    # Skip if an event with the same cat, amt, direct already exists on this date in relevant_events
                    exists = ((relevant_events['settlement_date'] == proj_date) &
                              (relevant_events['category'] == cat) &
                              (relevant_events['amount_home'] == amt) &
                              (relevant_events['direction'] == direct)).any()

                    if not exists:
                        delta = -amt if direct == 'debit' else amt
                        # Create a direction-aware synthetic ID for the audit trace
                        synthetic_id = f"proj_{direct}_{cat}_{amt}_{proj_date}"
                        add_delta_fn(proj_date, delta, synthetic_id)

    def calculate_safe_today_capacity(self, request_date: date) -> float:
        """
        Calculates the max amount safe to pay on request_date.
        amount_safe_to_pay = max(0, min(daily_cushions))
        """
        trace = self.simulate(request_date)
        max_safe = max(0.0, trace.min_cushion)
        return max_safe

    def find_earliest_full_payment_date(self, amount: float, request_date: date) -> Optional[date]:
        """
        Iterates day-by-day to find the first date where a full payment of 'amount'
        is safe for the remaining projection window.
        """
        # Check every day in the 90-day window
        for d in range(90):
            test_date = request_date + timedelta(days=d)
            # Simulate paying 'amount' on test_date
            trace = self.simulate(request_date, additional_payments=[(test_date, amount)])
            if trace.min_cushion >= 0:
                return test_date
        return None
