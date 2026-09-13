from pydantic import BaseModel, Field, field_validator
from datetime import date
from typing import Optional, Literal
import re

class CleanedLedgerEntry(BaseModel):
    """Strict schema for a resolved financial event in the cleaned ledger."""
    event_id: str
    user_id: str
    category: str
    amount: Optional[float] = Field(None, description="Original amount in original currency")
    currency: Optional[str] = Field(None, description="Currency of the amount")
    amount_home: Optional[float] = Field(None, description="Amount normalized to home currency")
    settlement_date: Optional[date] = Field(None, description="Date of settlement")
    direction: Optional[Literal['debit', 'credit', 'non_cash']] = Field(None, description="Direction of cash flow")
    status: Optional[Literal['settled', 'pending', 'scheduled', 'cancelled', 'unrealized', 'failed']] = Field(None, description="Current status of the event")

    class Config:
        extra = "allow" # Allow other metadata from original CSV


class OutputRow(BaseModel):
    """
    Strict Pydantic schema for evaluation output row according to
    problem_statement.md and AGENTS.md §6.2.
    """
    request_id: str
    amount_safe_to_pay: float
    affordability_status: Literal["affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"]
    recommended_payment_method: Literal["full_payment", "partial_payment", "installments", "wait", "not_recommended"]
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str

    @field_validator('amount_safe_to_pay')
    def validate_amount_safe(cls, v):
        if v < 0:
            raise ValueError(f"amount_safe_to_pay cannot be negative: {v}")
        return round(float(v), 2)

    @field_validator('payment_plan')
    def validate_payment_plan(cls, v, info):
        if v == "none":
            return v
        # Pattern: YYYY-MM-DD:<amount> separated by |
        parts = v.split('|')
        for p in parts:
            if not re.match(r'^\d{4}-\d{2}-\d{2}:[0-9]+(\.[0-9]+)?$', p):
                raise ValueError(f"Invalid payment_plan entry: {p}")
        return v

    @field_validator('spending_changes_needed')
    def validate_spending_changes(cls, v):
        if v == "none":
            return v
        parts = v.split('|')
        if len(parts) > 3:
            raise ValueError(f"spending_changes_needed exceeds max 3 changes: {v}")
        for p in parts:
            if not (p.startswith('stop:') or p.startswith('reduce_to:')):
                raise ValueError(f"Invalid spending change entry: {p}")
        return v
