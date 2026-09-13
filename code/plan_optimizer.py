import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import date, timedelta
from itertools import combinations
from code.cashflow_simulator import CashFlowSimulator

logger = logging.getLogger(__name__)

class PlanOptimizer:
    """
    Phase 4: Multi-Strategy Exhaustive Combinatorial Plan Optimizer.
    Generates and evaluates all viable payment plans (full_payment, installments,
    partial_payment, wait, and spending reduction variants) using exhaustive
    combinatorial search over up to 3 flexible spending changes.
    """

    def __init__(self):
        pass

    def optimize(self, request_id: str, request_data: Dict[str, Any],
                 user_profile: Dict[str, Any], cleaned_ledger: pd.DataFrame,
                 payment_options: pd.DataFrame) -> Dict[str, Any]:
        """
        Finds all viable candidate plans and selects the optimal plan based on
        the deterministic 6-tier tie-breaking hierarchy.
        """
        # 1. Parse request parameters
        req_date = pd.to_datetime(request_data['request_date']).date()
        req_amt = float(request_data['requested_amount'])
        deadline = pd.to_datetime(request_data['desired_completion_date']).date()
        allows_partial = bool(request_data.get('allows_partial_payment', False))

        # 2. Base simulation (BEFORE optional spending changes)
        # amount_safe_to_pay: largest amount safe on request_date before spending changes, capped at requested_amount
        base_sim = CashFlowSimulator(user_profile, cleaned_ledger)
        safe_today_raw = base_sim.calculate_safe_today_capacity(req_date)
        amount_safe_to_pay = min(req_amt, max(0.0, safe_today_raw))
        earliest_date_for_full_payment = base_sim.find_earliest_full_payment_date(req_amt, req_date)

        # 3. User payment preferences
        raw_methods = user_profile.get('payment_methods_user_will_consider', '')
        if pd.isna(raw_methods) or not str(raw_methods).strip():
            considered_methods = set()
        else:
            considered_methods = set(str(raw_methods).split('|'))

        # Max installment months
        raw_max_months = user_profile.get('max_installment_months')
        if pd.isna(raw_max_months) or raw_max_months is None or str(raw_max_months).strip() == '':
            max_installment_months = None
        else:
            try:
                max_installment_months = float(raw_max_months)
            except (ValueError, TypeError):
                max_installment_months = None

        # 4. Filter payment options for this request and max_installment_months
        req_options = payment_options[payment_options['request_id'] == request_id].copy()
        if not req_options.empty and max_installment_months is not None:
            # An installment option duration in months must not exceed max_installment_months
            def option_duration_months(row):
                if row.get('payment_method') != 'installments':
                    return 0.0
                num_p = float(row.get('number_of_payments', 1))
                freq = float(row.get('payment_frequency_days', 30))
                # Total span in days: (num_p - 1) * freq
                # Convert to months conservatively (span / 30.0)
                span_days = max(0.0, (num_p - 1) * freq)
                return span_days / 30.0

            valid_mask = req_options.apply(lambda r: option_duration_months(r) <= max_installment_months, axis=1)
            req_options = req_options[valid_mask]
        elif max_installment_months is None:
            # User will not consider installments if max_installment_months is blank
            req_options = req_options[req_options['payment_method'] != 'installments']

        # 5. Identify flexible spending events
        flexible_events = self._get_flexible_events(user_profile, cleaned_ledger)

        # 6. Generate spending change combinations (none, 1, 2, or 3 changes)
        spending_combos = self._generate_spending_combinations(flexible_events)

        # 7. Exhaustive evaluation
        all_candidates = []

        for combo in spending_combos:
            # Build modified ledger for this spending change combo
            modified_ledger = self._apply_spending_changes(cleaned_ledger, combo)
            sim = CashFlowSimulator(user_profile, modified_ledger)

            # Strategy 1: Full Payment
            if 'full_payment' in considered_methods:
                plan = self._evaluate_full_payment(sim, req_date, req_amt)
                if plan:
                    plan['spending_changes'] = combo
                    all_candidates.append(plan)

            # Strategy 2: Installments
            if 'installments' in considered_methods and not req_options.empty:
                inst_opts = req_options[req_options['payment_method'] == 'installments']
                for _, opt in inst_opts.iterrows():
                    plan = self._evaluate_installment_option(sim, req_date, req_amt, opt)
                    if plan:
                        plan['spending_changes'] = combo
                        all_candidates.append(plan)

            # Strategy 3: Partial Payment
            # Rule: allows_partial_payment is true, user considers partial_payment,
            # 0 < amount_safe_to_pay < requested_amount, and second payment on or before desired_completion_date.
            if ('partial_payment' in considered_methods and allows_partial and
                    0 < amount_safe_to_pay < req_amt and earliest_date_for_full_payment is not None and
                    earliest_date_for_full_payment <= deadline):
                plan = self._evaluate_partial_payment(sim, req_date, req_amt, deadline,
                                                      amount_safe_to_pay, earliest_date_for_full_payment)
                if plan:
                    plan['spending_changes'] = combo
                    all_candidates.append(plan)

            # Strategy 4: Wait
            # Rule: user considers full_payment, earliest_date_for_full_payment is valid and <= deadline
            if ('full_payment' in considered_methods and earliest_date_for_full_payment is not None and
                    earliest_date_for_full_payment <= deadline):
                plan = self._evaluate_wait(sim, req_date, req_amt, deadline, earliest_date_for_full_payment)
                if plan:
                    plan['spending_changes'] = combo
                    all_candidates.append(plan)

        # 8. Deterministic 6-tier ranking
        best_plan = None
        best_rank = None

        for cand in all_candidates:
            rank = self._calculate_rank(cand, deadline)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_plan = cand

        # 9. Fallback if no safe plan found
        if best_plan is None:
            return self._map_to_output(
                request_id=request_id,
                method="not_recommended",
                payments=None,
                changes=None,
                status="not_affordable",
                amount_safe_to_pay=amount_safe_to_pay,
                earliest_date_for_full_payment=earliest_date_for_full_payment
            )

        # Determine affordability status
        status = "affordable_with_plan"
        if best_plan['method'] == 'full_payment' and best_plan['first_payment_date'] == req_date and not best_plan['spending_changes']:
            status = "affordable_now"
        elif best_plan['method'] == 'wait':
            status = "affordable_later"

        return self._map_to_output(
            request_id=request_id,
            method=best_plan['method'],
            payments=best_plan['payments'],
            changes=best_plan['spending_changes'],
            status=status,
            amount_safe_to_pay=amount_safe_to_pay,
            earliest_date_for_full_payment=earliest_date_for_full_payment
        )

    def _get_flexible_events(self, user_profile: Dict[str, Any], ledger: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identifies recurring events that are permitted to be stopped or reduced
        by both dataset flexibility and user profile permissions.
        Only considers the latest event per category to avoid redundant combinations.
        """
        uid = str(user_profile.get('user_id'))
        user_ledger = ledger[ledger['user_id'].astype(str) == uid].copy()

        # Parse user permissions
        raw_stop = user_profile.get('expense_categories_user_is_willing_to_stop', '')
        stop_cats = set(str(raw_stop).split('|')) if pd.notna(raw_stop) and str(raw_stop).strip() else set()
        stop_cats.discard('')

        raw_reduce = user_profile.get('expense_categories_user_is_willing_to_reduce', '')
        reduce_cats = set(str(raw_reduce).split('|')) if pd.notna(raw_reduce) and str(raw_reduce).strip() else set()
        reduce_cats.discard('')

        # Categories protected from any changes
        raw_prot = user_profile.get('expense_categories_to_protect', '')
        prot_cats = set(str(raw_prot).split('|')) if pd.notna(raw_prot) and str(raw_prot).strip() else set()
        prot_cats.discard('')

        # Effective permissions
        stop_cats = stop_cats - prot_cats
        reduce_cats = reduce_cats - prot_cats

        flexible_candidates = []
        for _, event in user_ledger.iterrows():
            if event.get('direction') != 'debit':
                continue

            cat = event.get('category')
            flex = event.get('flexibility', 'fixed')
            eid = event.get('event_id')
            amt = float(event.get('amount_home', 0.0))
            min_allowed = event.get('minimum_allowed_amount')

            # Can stop? Event must be stoppable AND user permits stopping category
            can_stop = (flex in ['stoppable', 'reducible_or_stoppable']) and (cat in stop_cats)

            # Can reduce? Event must be reducible AND user permits reducing category AND minimum_allowed_amount is present
            can_reduce = (flex in ['reducible', 'reducible_or_stoppable']) and (cat in reduce_cats) and (pd.notna(min_allowed))

            if can_stop or can_reduce:
                min_val = float(min_allowed) if pd.notna(min_allowed) else 0.0
                # Reduction is only meaningful if min_val < amt
                can_reduce = can_reduce and (min_val < amt)
                if can_stop or can_reduce:
                    flexible_candidates.append({
                        'event_id': str(eid),
                        'category': cat,
                        'amount': amt,
                        'settlement_date': str(event.get('settlement_date')),
                        'stoppable': can_stop,
                        'reducible': can_reduce,
                        'min_allowed': min_val
                    })

        if not flexible_candidates:
            return []

        # Group by category and keep the most recent event to represent the recurring commitment
        cand_df = pd.DataFrame(flexible_candidates)
        cand_df = cand_df.sort_values('settlement_date', ascending=False)
        deduped = cand_df.drop_duplicates(subset=['category'], keep='first')
        return deduped.to_dict('records')

    def _generate_spending_combinations(self, flexible_events: List[Dict[str, Any]]) -> List[List[str]]:
        """
        Generates exhaustive combinations of up to 3 spending changes.
        Mutually exclusive: never both stop and reduce_to on the same event_id.
        """
        # Possible single changes:
        atomic_actions = []
        for ev in flexible_events:
            eid = ev['event_id']
            if ev['stoppable']:
                atomic_actions.append(f"stop:{eid}")
            if ev['reducible']:
                # Format integer if exact, else 2 decimal places
                m = ev['min_allowed']
                amt_str = f"{int(m)}" if m == int(m) else f"{m:.2f}"
                atomic_actions.append(f"reduce_to:{eid}:{amt_str}")

        valid_combos = [[]] # Empty changes (none) is always the first candidate

        # Helper to check mutual exclusivity on the same event
        def is_valid_subset(subset):
            event_ids = set()
            for action in subset:
                eid = action.split(':')[1]
                if eid in event_ids:
                    return False
                event_ids.add(eid)
            return True

        # Generate subsets of size 1, 2, 3
        for k in range(1, min(4, len(atomic_actions) + 1)):
            for subset in combinations(atomic_actions, k):
                if is_valid_subset(subset):
                    valid_combos.append(list(subset))

        return valid_combos

    def _apply_spending_changes(self, ledger: pd.DataFrame, changes: List[str]) -> pd.DataFrame:
        """
        Applies spending changes to the ledger.
        If an event is stopped or reduced, adjusts all events in that category
        with the matching historical amount to ensure projected recurrence reflects the change.
        """
        if not changes:
            return ledger

        df = ledger.copy()
        for change in changes:
            if change.startswith('stop:'):
                eid = change.split(':')[1]
                target = df[df['event_id'] == eid]
                if not target.empty:
                    cat = target.iloc[0]['category']
                    amt = target.iloc[0]['amount_home']
                    # Zero out this event and matching historical recurring occurrences in category
                    df.loc[(df['category'] == cat) & (df['amount_home'] == amt), 'amount_home'] = 0.0
                else:
                    df.loc[df['event_id'] == eid, 'amount_home'] = 0.0

            elif change.startswith('reduce_to:'):
                parts = change.split(':')
                eid = parts[1]
                new_amt = float(parts[2])
                target = df[df['event_id'] == eid]
                if not target.empty:
                    cat = target.iloc[0]['category']
                    orig_amt = target.iloc[0]['amount_home']
                    # Reduce this event and matching historical recurring occurrences in category
                    df.loc[(df['category'] == cat) & (df['amount_home'] == orig_amt), 'amount_home'] = new_amt
                else:
                    df.loc[df['event_id'] == eid, 'amount_home'] = new_amt

        return df

    def _evaluate_full_payment(self, sim: CashFlowSimulator, req_date: date, req_amt: float) -> Optional[Dict]:
        """Evaluates paying in full on request_date."""
        trace = sim.simulate(req_date, additional_payments=[(req_date, req_amt)])
        if trace.min_cushion >= 0:
            return {
                'method': 'full_payment',
                'payments': [(req_date, req_amt)],
                'total_amount': req_amt,
                'completion_date': req_date,
                'first_payment_date': req_date,
                'num_payments': 1,
                'payment_option_id': None
            }
        return None

    def _evaluate_installment_option(self, sim: CashFlowSimulator, req_date: date, req_amt: float, opt: pd.Series) -> Optional[Dict]:
        """Evaluates an exact installment option from request_payment_options.csv."""
        first_date = pd.to_datetime(opt['first_payment_date']).date()
        num_pay = int(opt['number_of_payments'])
        freq = int(opt['payment_frequency_days'])
        total = float(opt['total_payable_amount'])
        pay_amt = float(opt['payment_amount'])

        payments = []
        for i in range(num_pay):
            p_date = first_date + timedelta(days=i * freq)
            payments.append((p_date, pay_amt))

        trace = sim.simulate(req_date, additional_payments=payments)
        if trace.min_cushion >= 0:
            return {
                'method': 'installments',
                'payments': payments,
                'total_amount': total,
                'completion_date': payments[-1][0],
                'first_payment_date': payments[0][0],
                'num_payments': num_pay,
                'payment_option_id': str(opt['payment_option_id'])
            }
        return None

    def _evaluate_partial_payment(self, sim: CashFlowSimulator, req_date: date, req_amt: float, deadline: date,
                                  amount_safe_to_pay: float, earliest_date_for_full_payment: date) -> Optional[Dict]:
        """
        Evaluates partial payment: exactly two payments summing to requested_amount.
        Payment 1: amount_safe_to_pay on request_date.
        Payment 2: (req_amt - amount_safe_to_pay) on earliest_date_for_full_payment.
        """
        p2_amt = req_amt - amount_safe_to_pay
        payments = [
            (req_date, amount_safe_to_pay),
            (earliest_date_for_full_payment, p2_amt)
        ]

        trace = sim.simulate(req_date, additional_payments=payments)
        if trace.min_cushion >= 0:
            return {
                'method': 'partial_payment',
                'payments': payments,
                'total_amount': req_amt,
                'completion_date': earliest_date_for_full_payment,
                'first_payment_date': req_date,
                'num_payments': 2,
                'payment_option_id': None
            }
        return None

    def _evaluate_wait(self, sim: CashFlowSimulator, req_date: date, req_amt: float, deadline: date,
                       earliest_date_for_full_payment: date) -> Optional[Dict]:
        """Evaluates waiting until earliest_date_for_full_payment for full payment."""
        payments = [(earliest_date_for_full_payment, req_amt)]
        trace = sim.simulate(req_date, additional_payments=payments)
        if trace.min_cushion >= 0:
            return {
                'method': 'wait',
                'payments': payments,
                'total_amount': req_amt,
                'completion_date': earliest_date_for_full_payment,
                'first_payment_date': earliest_date_for_full_payment,
                'num_payments': 1,
                'payment_option_id': None
            }
        return None

    def _calculate_rank(self, plan: Dict, deadline: date) -> Tuple:
        """
        Deterministic 6-tier ranking key according to problem_statement.md:
        1. Complete the full request by desired_completion_date.
        2. Require no spending changes.
        3. Minimize the total amount paid.
        4. Start payment earlier.
        5. Use fewer payments.
        6. Use the lowest payment_option_id (safely as string/tuple).
        """
        tier1 = 0 if plan['completion_date'] <= deadline else 1
        tier2 = 0 if not plan['spending_changes'] else 1
        tier3 = plan['total_amount']
        tier4 = plan['first_payment_date'].toordinal()
        tier5 = plan['num_payments']
        # Tier 6: Safe comparison for payment_option_id without assumption of integer format
        opt_id = plan.get('payment_option_id')
        tier6 = opt_id if opt_id is not None else "zzzzzzzzzz"

        return (tier1, tier2, tier3, tier4, tier5, tier6)

    def _map_to_output(self, request_id: str, method: str, payments: Optional[List[Tuple[date, float]]],
                       changes: Optional[List[str]], status: str,
                       amount_safe_to_pay: float, earliest_date_for_full_payment: Optional[date]) -> Dict[str, Any]:
        """Formats the result according to problem_statement.md output contract."""
        # payment_plan: <YYYY-MM-DD>:<amount>|<YYYY-MM-DD>:<amount> or "none"
        if not payments:
            plan_str = "none"
        else:
            fmt_parts = []
            for d, a in payments:
                if a == int(a):
                    fmt_parts.append(f"{d}:{int(a)}")
                else:
                    s = f"{a:.2f}"
                    if '.' in s:
                        s = s.rstrip('0').rstrip('.')
                    fmt_parts.append(f"{d}:{s}")
            plan_str = "|".join(fmt_parts)

        # spending_changes_needed: up to 3 changes separated by | or "none"
        if not changes:
            changes_str = "none"
        else:
            changes_str = "|".join(changes)

        # earliest_date_for_full_payment: equals request_date for affordable_now, empty string if None
        if status == "affordable_now":
            full_date_str = payments[0][0].strftime("%Y-%m-%d") if payments else ""
        elif earliest_date_for_full_payment is not None:
            full_date_str = earliest_date_for_full_payment.strftime("%Y-%m-%d")
        else:
            full_date_str = ""

        return {
            'request_id': request_id,
            'amount_safe_to_pay': amount_safe_to_pay,
            'affordability_status': status,
            'recommended_payment_method': method,
            'payment_plan': plan_str,
            'earliest_date_for_full_payment': full_date_str,
            'spending_changes_needed': changes_str,
            'decision_explanation': ""
        }
