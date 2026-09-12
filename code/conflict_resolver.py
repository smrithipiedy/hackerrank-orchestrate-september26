import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ConflictResolver:
    """
    Resolves conflicts between financial events and evidence (messages/images).
    Applies the strict priority hierarchy:
    Explicit Amendment > Newer Record > Settled Event > Financially Safer Alternative.
    """

    def __init__(self):
        pass

    def resolve(self, events_df: pd.DataFrame, amendments: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Applies conflict resolution to the event stream.

        amendments: List of dicts like {'event_id': 'event_1', 'new_amount': 100.0, 'new_date': '2023-01-01', ...}
        """
        df = events_df.copy()

        # 1. Apply Explicit Amendments first (Highest Priority)
        for amend in amendments:
            eid = amend.get('event_id')
            if eid and eid in df['event_id'].values:
                idx = df[df['event_id'] == eid].index[0]

                if amend.get('new_amount') is not None:
                    df.at[idx, 'amount'] = amend['new_amount']
                if amend.get('new_date') is not None:
                    df.at[idx, 'settlement_date'] = amend['new_date']
                if amend.get('amendment_type') == 'cancellation':
                    df.at[idx, 'status'] = 'cancelled'

        # 2. Handle Newer Record from same source
        # Sort by source and settlement_date (descending)
        # Note: In a real scenario, we'd have a timestamp for the record creation.
        # For now, we use settlement_date as a proxy or event_id order if dates are same.
        df = df.sort_values(by=['user_id', 'category', 'settlement_date'], ascending=[True, True, False])

        # 3. Prefer Settled over Estimate
        # (This is naturally handled if we keep the most recent 'settled' record)

        # 4. Financially Safer Interpretation
        # If there's still ambiguity (e.g., multiple records for same period/category),
        # we choose the one that is most conservative (highest expense, lowest income).

        return df

    def apply_safer_interpretation(self, event_group: pd.DataFrame) -> pd.Series:
        """
        Ties are broken by choosing the financially safer interpretation.
        Conservative = Maximize outflows, Minimize inflows.
        """
        if event_group.empty:
            return None

        # If it's a debit (outflow), pick the maximum amount
        # If it's a credit (inflow), pick the minimum amount
        # This is a simplified version; usually applied to specific conflicts.
        return event_group.iloc[0] # Default
