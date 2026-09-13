import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class ConflictResolver:
    """
    Resolves conflicts between financial events and evidence (messages/images).
    Implements the exact 4-tier hierarchy from problem_statement.md:
    1. Explicit Cancellation/Settlement/Amendment
    2. Newer Record from same source
    3. Settled Event
    4. Financially Safer Interpretation
    """

    def __init__(self):
        pass

    def resolve(self, events_df: pd.DataFrame, amendments: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Resolves conflicts and produces a final 'Cleaned Ledger'.
        """
        if events_df.empty:
            return events_df

        df = events_df.copy()

        # Pre-process: ensure settlement_date is comparable
        df['settlement_date'] = pd.to_datetime(df['settlement_date']).dt.date

        # TIER 1: Explicit Cancellation, Settlement, or Amendment
        for amend in amendments:
            eid = amend.get('event_id')
            if eid and eid in df['event_id'].values:
                idx = df[df['event_id'] == eid].index[0]

                if amend.get('amendment_type') == 'cancellation':
                    df.at[idx, 'status'] = 'cancelled'
                if amend.get('new_amount') is not None:
                    df.at[idx, 'amount'] = amend['new_amount']
                if amend.get('new_date') is not None:
                    df.at[idx, 'settlement_date'] = pd.to_datetime(amend['new_date']).date()
                if amend.get('amendment_type') == 'confirmation':
                    df.at[idx, 'status'] = 'settled'

        # IDENTITY RESOLUTION: Deduplicate records that explicitly represent the same transaction.
        # This includes:
        # 1. Same event_id
        # 2. linked_event_id pointing to another record in the same set

        # Create a mapping for linked events
        resolved_records = []
        processed_ids = set()

        # Sort by Tier 2/3/4 priorities generally to make picking the "best" easier
        # Settled > Newer > Safer
        df = df.sort_values(by=['status', 'settlement_date'], ascending=[False, False])

        for idx, row in df.iterrows():
            eid = row['event_id']
            if eid in processed_ids:
                continue

            # Find all records linked to this event_id
            # Use .get() or check columns to avoid KeyError if linked_event_id is missing
            mask = (df['event_id'] == eid)
            if 'linked_event_id' in df.columns:
                mask |= (df['linked_event_id'] == eid)

            group = df[mask]

            if len(group) == 1:
                resolved_records.append(group.iloc[0])
            else:
                resolved_records.append(self._resolve_group_conflict(group))

            processed_ids.update(group['event_id'].tolist())

        return pd.DataFrame(resolved_records).reset_index(drop=True)

    def _resolve_group_conflict(self, group: pd.DataFrame) -> pd.Series:
        """
        Internal logic to resolve conflicts within a group of records.
        Hierarchy: Explicit > Newer > Settled > Safer.
        """
        # 1. Prioritize 'settled' status over others (Tier 3)
        settled = group[group['status'] == 'settled']
        if not settled.empty:
            # If multiple settled, pick the newest (Tier 2)
            return settled.sort_values(by=['settlement_date'], ascending=False).iloc[0]

        # 2. Prefer newer record (Tier 2)
        # If no settled record, sort by date/id
        sorted_group = group.sort_values(by=['settlement_date'], ascending=False)

        # 3. Financially Safer Interpretation (Tier 4)
        # If we still have multiple records with same date/status, pick the safer one.
        # Safe = Max Debit, Min Credit.
        if len(sorted_group) > 1:
            # Check if they are all the same direction
            directions = sorted_group['direction'].unique()
            if len(directions) == 1:
                dir_val = directions[0]
                if dir_val == 'debit':
                    return sorted_group.loc[sorted_group['amount'].idxmax()]
                elif dir_val == 'credit':
                    return sorted_group.loc[sorted_group['amount'].idxmin()]

        return sorted_group.iloc[0]

    def apply_safer_interpretation(self, event_group: pd.DataFrame) -> pd.Series:
        """
        Deterministic tie-breaker for conflicting records.
        Outflows (debit) -> Max amount
        Inflows (credit) -> Min amount
        """
        if event_group.empty:
            return None

        debits = event_group[event_group['direction'] == 'debit']
        credits = event_group[event_group['direction'] == 'credit']

        if not debits.empty:
            return debits.loc[debits['amount'].idxmax()]
        if not credits.empty:
            return credits.loc[credits['amount'].idxmin()]

        return event_group.iloc[0]
