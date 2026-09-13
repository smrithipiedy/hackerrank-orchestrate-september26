import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import date
import pandas as pd
from code.explanation_generator import ExplanationGenerator

logger = logging.getLogger(__name__)

class DecisionEngine:
    """
    Phase 5: Deterministic 6-Tier Decision Engine & Multi-Candidate Selector.
    Consumes candidate plans from Phase 4 Combinatorial Plan Optimizer,
    enforces payment-method eligibility/preferences, applies the strict
    6-tier ranking defined in problem_statement.md, handles safe fallback,
    and formats the final decision fields according to the project contract.
    Phase 6: Integrates grounded explanation generation.
    """

    def __init__(self):
        self.explainer = ExplanationGenerator()

    def select_best_plan(self, candidates: List[Dict[str, Any]], deadline: date) -> Optional[Dict[str, Any]]:
        """
        Applies the exact 6-tier ranking hierarchy to select the single best plan:
        1. Complete the full request by desired_completion_date.
        2. Require no spending changes.
        3. Minimize the total amount paid.
        4. Start payment earlier.
        5. Use fewer payments.
        6. Use the lowest payment_option_id as the final tie-breaker (safe string comparison).
        """
        if not candidates:
            return None

        def plan_ranking_key(plan: Dict[str, Any]) -> Tuple:
            tier1 = 0 if plan['completion_date'] <= deadline else 1
            tier2 = 0 if not plan.get('spending_changes') else 1
            tier3 = float(plan['total_amount'])
            tier4 = plan['first_payment_date'].toordinal()
            tier5 = int(plan['num_payments'])
            opt_id = plan.get('payment_option_id')
            tier6 = str(opt_id) if opt_id is not None else "zzzzzzzzzz"
            return (tier1, tier2, tier3, tier4, tier5, tier6)

        return min(candidates, key=plan_ranking_key)

    def format_decision(self, request_id: str, best_plan: Optional[Dict[str, Any]],
                        req_date: date, amount_safe_to_pay: float,
                        earliest_date_for_full_payment: Optional[date],
                        explanation: str = "") -> Dict[str, Any]:
        """
        Formats the final decision according to the dataset/output.csv contract:
        - request_id
        - amount_safe_to_pay
        - affordability_status
        - recommended_payment_method
        - payment_plan
        - earliest_date_for_full_payment
        - spending_changes_needed
        - decision_explanation
        """
        if best_plan is None:
            # Fallback when no safe eligible plan is found
            status = "not_affordable"
            method = "not_recommended"
            plan_str = "none"
            changes_str = "none"
            full_date_str = earliest_date_for_full_payment.strftime("%Y-%m-%d") if earliest_date_for_full_payment is not None else ""
        else:
            method = best_plan['method']
            changes = best_plan.get('spending_changes')
            changes_str = "|".join(changes) if changes else "none"

            # Format payment_plan string
            payments = best_plan.get('payments', [])
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

            # Determine affordability status
            if method == 'full_payment' and best_plan['first_payment_date'] == req_date and not changes:
                status = "affordable_now"
            elif method == 'wait':
                status = "affordable_later"
            else:
                status = "affordable_with_plan"

            # earliest_date_for_full_payment:
            # equals request_date for affordable_now
            if status == "affordable_now":
                full_date_str = payments[0][0].strftime("%Y-%m-%d") if payments else req_date.strftime("%Y-%m-%d")
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
            'decision_explanation': explanation
        }

    def evaluate_request(self, request_id: str, request_data: Dict[str, Any],
                         user_profile: Dict[str, Any], cleaned_ledger: pd.DataFrame,
                         payment_options: pd.DataFrame,
                         optimizer: Optional[Any] = None) -> Dict[str, Any]:
        """
        Orchestrates candidate generation via PlanOptimizer, selects the optimal plan
        using 6-tier deterministic ranking, synthesizes the grounded explanation,
        and formats the output dictionary.
        """
        if optimizer is None:
            from code.plan_optimizer import PlanOptimizer
            optimizer = PlanOptimizer()

        # 1. Base simulation for safe today & earliest full payment date
        req_date = pd.to_datetime(request_data['request_date']).date()
        req_amt = float(request_data['requested_amount'])
        deadline = pd.to_datetime(request_data['desired_completion_date']).date()

        from code.cashflow_simulator import CashFlowSimulator
        base_sim = CashFlowSimulator(user_profile, cleaned_ledger)
        safe_today_raw = base_sim.calculate_safe_today_capacity(req_date)
        amount_safe_to_pay = min(req_amt, max(0.0, safe_today_raw))
        earliest_date_for_full_payment = base_sim.find_earliest_full_payment_date(req_amt, req_date)

        # 2. Optimize plan
        opt_res = optimizer.optimize(
            request_id=request_id,
            request_data=request_data,
            user_profile=user_profile,
            cleaned_ledger=cleaned_ledger,
            payment_options=payment_options
        )

        # 3. Best plan details reconstructed for explanation synthesis
        best_plan = None
        method = opt_res['recommended_payment_method']
        if method != 'not_recommended':
            # Parse payments from opt_res['payment_plan']
            payments = []
            if opt_res['payment_plan'] != 'none':
                for part in opt_res['payment_plan'].split('|'):
                    d_str, a_str = part.split(':')
                    payments.append((pd.to_datetime(d_str).date(), float(a_str)))

            best_plan = {
                'method': method,
                'payments': payments,
                'total_amount': sum(a for _, a in payments) if payments else req_amt,
                'spending_changes': [] if opt_res['spending_changes_needed'] == 'none' else opt_res['spending_changes_needed'].split('|'),
                'payment_option_id': None
            }

        # 4. Synthesize explanation
        explanation = self.explainer.generate_explanation(
            request_data=request_data,
            user_profile=user_profile,
            decision=opt_res,
            best_plan=best_plan
        )
        opt_res['decision_explanation'] = explanation

        return opt_res
