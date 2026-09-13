#!/usr/bin/env python3
"""
verify_submission.py
Final verification script according to problem_statement.md and AGENTS.md.
Validates:
1. output.csv exists, has exactly 250 rows matching dataset/requests.csv
2. Required 8 columns in exact order
3. Amount bounds: 0 <= amount_safe_to_pay <= requested_amount
4. Schema validation against OutputRow Pydantic model
5. Deliverables check: code.zip, output.csv, evaluation/usage_report.md
"""

import os
import sys
import zipfile
import pandas as pd
from code.schemas import OutputRow

def verify_output_file(path: str, req_df: pd.DataFrame) -> bool:
    print(f"--- Verifying {path} ---")
    if not os.path.exists(path):
        print(f"FAIL: File does not exist: {path}")
        return False

    df = pd.read_csv(path)
    print(f"Row count: {len(df)}")
    if len(df) != 250:
        print(f"FAIL: Expected 250 rows, got {len(df)}")
        return False

    required_cols = [
        'request_id',
        'amount_safe_to_pay',
        'affordability_status',
        'recommended_payment_method',
        'payment_plan',
        'earliest_date_for_full_payment',
        'spending_changes_needed',
        'decision_explanation'
    ]

    if list(df.columns) != required_cols:
        print(f"FAIL: Columns mismatch.\nExpected: {required_cols}\nGot: {list(df.columns)}")
        return False

    if (df['request_id'] != req_df['request_id']).any():
        print("FAIL: request_id column does not match dataset/requests.csv exactly in order.")
        return False

    # Bounds check
    merged = pd.merge(df, req_df, on='request_id')
    for idx, row in merged.iterrows():
        safe_amt = float(row['amount_safe_to_pay'])
        req_amt = float(row['requested_amount'])
        if safe_amt < 0.0 or safe_amt > (req_amt + 1e-5):
            print(f"FAIL: Bounds error on request {row['request_id']}: safe {safe_amt} vs requested {req_amt}")
            return False

    # Schema check
    validation_errors = 0
    for idx, row in df.iterrows():
        try:
            row_dict = row.to_dict()
            if pd.isna(row_dict.get('earliest_date_for_full_payment')):
                row_dict['earliest_date_for_full_payment'] = ''
            OutputRow.model_validate(row_dict)
        except Exception as e:
            print(f"FAIL: Schema error on row {idx} ({row['request_id']}): {e}")
            validation_errors += 1
            if validation_errors > 5:
                break

    if validation_errors > 0:
        print(f"FAIL: Total schema errors: {validation_errors}")
        return False

    print(f"PASS: {path} is 100% compliant with challenge contract.")
    return True

def verify_code_zip(path: str) -> bool:
    print(f"--- Verifying {path} ---")
    if not os.path.exists(path):
        print(f"FAIL: Archive does not exist: {path}")
        return False

    try:
        with zipfile.ZipFile(path, 'r') as z:
            namelist = z.namelist()
            print(f"Found {len(namelist)} items in {path}.")

            # Must contain evaluation/usage_report.md
            has_usage = any('evaluation/usage_report.md' in n for n in namelist)
            if not has_usage:
                print("FAIL: code.zip is missing evaluation/usage_report.md")
                return False

            # Must not contain __pycache__ or .cache or secrets
            bad_patterns = ['__pycache__', '.env', '.git/', '.cache/']
            for item in namelist:
                for bp in bad_patterns:
                    if bp in item:
                        print(f"FAIL: code.zip contains forbidden development item: {item}")
                        return False

        print(f"PASS: {path} is valid and contains required submission deliverables.")
        return True
    except Exception as e:
        print(f"FAIL: Error reading {path}: {e}")
        return False

def main():
    print("========================================")
    print("HackerRank Orchestrate: Final Submission Verification")
    print("========================================")

    req_df = pd.read_csv('dataset/requests.csv')

    # 1. Verify dataset/output.csv
    v1 = verify_output_file('dataset/output.csv', req_df)

    # 2. Verify root output.csv
    v2 = verify_output_file('output.csv', req_df)

    # 3. Verify code.zip (if present)
    v3 = True
    if os.path.exists('code.zip'):
        v3 = verify_code_zip('code.zip')
    else:
        print("INFO: code.zip not yet created.")

    if v1 and v2 and v3:
        print("\nOVERALL STATUS: ALL SUBMISSION CHECKS PASSED [OK]")
        sys.exit(0)
    else:
        print("\nOVERALL STATUS: VERIFICATION FAILED [FAIL]")
        sys.exit(1)

if __name__ == "__main__":
    main()
