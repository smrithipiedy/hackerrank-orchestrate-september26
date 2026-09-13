import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import date

logger = logging.getLogger(__name__)

class ExplanationGenerator:
    """
    Phase 6: Grounded Explanation Generator.
    Produces concise, grounded decision explanations citing the user's home currency,
    exact amounts, dates, and safety cushion/minimum balance constraints.
    Zero hallucinated financial facts; 100% grounded in simulation data.
    """

    def __init__(self):
        pass

    def _fmt_money(self, amount: float, currency: str) -> str:
        """Formats amount with currency, integer if whole number, else 2 decimal places."""
        if amount == int(amount):
            val_str = f"{int(amount):,}"
        else:
            val_str = f"{amount:,.2f}"
        return f"{currency} {val_str}"

    def _fmt_date(self, d: date) -> str:
        """Formats date as 'D Month YYYY', e.g. '15 November 2019'."""
        return f"{d.day} {d.strftime('%B')} {d.year}"

    def generate_explanation(self, request_data: Dict[str, Any], user_profile: Dict[str, Any],
                             decision: Dict[str, Any], best_plan: Optional[Dict[str, Any]],
                             min_cushion: float = 0.0) -> str:
        """
        Synthesizes a grounded explanation according to the challenge decision style:
        - full_payment (affordable_now):
            "Pay [Curr] [Amt] today. This leaves at least [Curr] [Min/Buffer] available over the next 90 days."
        - full_payment (affordable_with_plan):
            "[Stop/Reduce actions], then pay [Curr] [Amt] today. This leaves at least [Curr] [Min/Buffer] available."
        - installments (affordable_with_plan):
            "Use [N] installments of [Curr] [Amt], starting [Date]. This leaves at least [Curr] [Min/Buffer] available."
        - partial_payment (affordable_with_plan):
            "Pay [Curr] [Amt1] today and the remaining [Curr] [Amt2] on [Date2]. This completes the full request and keeps the [Curr] [Min] minimum protected."
        - wait (affordable_later):
            "Wait until [Date], then pay [Curr] [Amt] in full. Paying sooner would put the [Curr] [Min] minimum at risk."
        - not_recommended (not_affordable):
            "Do not make this payment by [Deadline]. None of the available options keeps the [Curr] [Min] minimum protected."
        """
        currency = user_profile.get('home_currency', 'USD')
        req_amt = float(request_data.get('requested_amount', 0.0))
        req_date = pd_to_date(request_data.get('request_date'))
        deadline = pd_to_date(request_data.get('desired_completion_date'))
        min_balance = float(user_profile.get('minimum_balance_to_keep', 0.0))

        method = decision.get('recommended_payment_method')
        status = decision.get('affordability_status')
        changes_str = decision.get('spending_changes_needed', 'none')

        # Fallback explanation
        if method == "not_recommended" or status == "not_affordable":
            deadline_str = self._fmt_date(deadline) if deadline else "the desired date"
            min_bal_str = self._fmt_money(min_balance, currency)
            safe_today = float(decision.get('amount_safe_to_pay', 0.0))
            # If some cash safe today but full cannot be completed:
            if safe_today > 0 and safe_today < req_amt and not decision.get('earliest_date_for_full_payment'):
                return f"Do not proceed with the {self._fmt_money(req_amt, currency)} request. Although {self._fmt_money(safe_today, currency)} is available today, the full amount cannot be completed safely within 90 days."
            return f"Do not make this payment by {deadline_str}. None of the available options keeps the {min_bal_str} minimum protected."

        # Strategy 1: full_payment
        if method == "full_payment":
            amt_str = self._fmt_money(req_amt, currency)
            min_bal_str = self._fmt_money(min_balance, currency)
            if status == "affordable_now":
                return f"Pay {amt_str} today. This leaves at least {min_bal_str} available over the next 90 days."
            else:
                # affordable_with_plan: spending changes required
                action_phrase = self._describe_spending_changes(changes_str, currency)
                return f"{action_phrase}, then pay {amt_str} today. This leaves at least {min_bal_str} available."

        # Strategy 2: installments
        if method == "installments":
            payments = best_plan.get('payments', []) if best_plan else []
            num_p = len(payments)
            first_pay_date = payments[0][0] if payments else req_date
            each_amt = payments[0][1] if payments else (req_amt / num_p if num_p else req_amt)
            date_str = self._fmt_date(first_pay_date)
            min_bal_str = self._fmt_money(min_balance, currency)
            return f"Use {num_p} installments of {self._fmt_money(each_amt, currency)}, starting {date_str}. This leaves at least {min_bal_str} available."

        # Strategy 3: partial_payment
        if method == "partial_payment":
            payments = best_plan.get('payments', []) if best_plan else []
            if len(payments) >= 2:
                amt1 = payments[0][1]
                amt2 = payments[1][1]
                date2_str = self._fmt_date(payments[1][0])
            else:
                amt1 = float(decision.get('amount_safe_to_pay', 0.0))
                amt2 = req_amt - amt1
                full_d = pd_to_date(decision.get('earliest_date_for_full_payment'))
                date2_str = self._fmt_date(full_d) if full_d else "later"

            min_bal_str = self._fmt_money(min_balance, currency)
            return f"Pay {self._fmt_money(amt1, currency)} today and the remaining {self._fmt_money(amt2, currency)} on {date2_str}. This completes the full request and keeps the {min_bal_str} minimum protected."

        # Strategy 4: wait
        if method == "wait":
            payments = best_plan.get('payments', []) if best_plan else []
            pay_date = payments[0][0] if payments else pd_to_date(decision.get('earliest_date_for_full_payment'))
            date_str = self._fmt_date(pay_date) if pay_date else "later"
            amt_str = self._fmt_money(req_amt, currency)
            min_bal_str = self._fmt_money(min_balance, currency)
            return f"Wait until {date_str}, then pay {amt_str} in full. Paying sooner would put the {min_bal_str} minimum at risk."

        return f"Proceed with {method} plan of {self._fmt_money(req_amt, currency)}."

    def _describe_spending_changes(self, changes_str: str, currency: str) -> str:
        """Converts change tokens into natural phrase, e.g. 'Stop event_14 and reduce event_21 to USD 50'."""
        if not changes_str or changes_str == "none":
            return "Adjust flexible spending"
        tokens = changes_str.split('|')
        phrased = []
        for t in tokens:
            if t.startswith('stop:'):
                eid = t.split(':')[1]
                phrased.append(f"stop {eid}")
            elif t.startswith('reduce_to:'):
                parts = t.split(':')
                eid = parts[1]
                amt = float(parts[2])
                phrased.append(f"reduce {eid} to {self._fmt_money(amt, currency)}")
        if len(phrased) == 1:
            return phrased[0].capitalize()
        return (", ".join(phrased[:-1]) + " and " + phrased[-1]).capitalize()


def pd_to_date(val: Any) -> Optional[date]:
    """Helper to convert string/Timestamp/date to datetime.date."""
    if val is None or val == "":
        return None
    if isinstance(val, date):
        return val
    import pandas as pd
    try:
        return pd.to_datetime(val).date()
    except Exception:
        return None
