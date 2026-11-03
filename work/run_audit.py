import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

# Ensure immediate unbuffered output
sys.stdout.reconfigure(line_buffering=True)

FILES = [
    'train_1.csv', 'train_2.csv', 'train_3.csv', 'train_4.csv', 'train_5.csv', 'train_6.csv',
    'val_7.csv', 'test_8.csv', 'test_9.csv', 'test_hard_10.csv'
]

ROLES = {
    'train_1.csv': 'train',
    'train_2.csv': 'train',
    'train_3.csv': 'train',
    'train_4.csv': 'train',
    'train_5.csv': 'train',
    'train_6.csv': 'val (internal validation split)',
    'val_7.csv': 'LLM data (eval)',
    'test_8.csv': 'test',
    'test_9.csv': 'test',
    'test_hard_10.csv': 'test_hard'
}

EXCLUDE_FROM_FEATURES = [
    'runId', 'simTime', 'receiverId', 'senderId', 'msgSeq',
    'label', 'attackType', 'attackerNode'
]

def audit():
    output_lines = []
    def log(msg=""):
        print(msg, flush=True)
        output_lines.append(str(msg))

    log("=" * 80)
    log("VANET INTRUSION DETECTION DATASET - COMPREHENSIVE PHASE 0 AUDIT REPORT")
    log("=" * 80)

    train_feat_by_attack = defaultdict(lambda: defaultdict(list))
    file_summaries = {}

    for fname in FILES:
        if not os.path.exists(fname):
            log(f"WARNING: File {fname} not found!")
            continue

        log(f"\n" + "-" * 80)
        log(f"FILE: {fname} (Role: {ROLES.get(fname, 'unknown')})")
        log("-" * 80)

        chunksize = 150_000
        total_rows = 0
        nan_counts = defaultdict(int)
        inf_counts = defaultdict(int)
        col_min = {}
        col_max = {}
        col_dtypes = {}
        label_counts = defaultdict(int)
        attack_counts = defaultdict(int)
        unique_senders = set()
        unique_receivers = set()
        pair_seq_lengths = defaultdict(int)
        time_min = float('inf')
        time_max = float('-inf')

        for chunk_idx, chunk in enumerate(pd.read_csv(fname, chunksize=chunksize, low_memory=False)):
            total_rows += len(chunk)
            if chunk_idx == 0:
                for col in chunk.columns:
                    col_dtypes[col] = str(chunk[col].dtype)

            # Check NaNs and Infs and min/max
            for col in chunk.columns:
                n_nan = int(chunk[col].isna().sum())
                if n_nan > 0:
                    nan_counts[col] += n_nan
                if pd.api.types.is_numeric_dtype(chunk[col]):
                    s = chunk[col].dropna()
                    if len(s) > 0:
                        n_inf = int(np.isinf(s).sum())
                        if n_inf > 0:
                            inf_counts[col] += n_inf
                        c_min = float(s.min())
                        c_max = float(s.max())
                        col_min[col] = min(col_min.get(col, float('inf')), c_min)
                        col_max[col] = max(col_max.get(col, float('-inf')), c_max)

            # Labels
            for val, cnt in chunk['label'].value_counts().items():
                label_counts[val] += cnt
            for val, cnt in chunk['attackType'].value_counts().items():
                attack_counts[val] += cnt

            # Senders and receivers
            unique_senders.update(chunk['senderId'].unique())
            unique_receivers.update(chunk['receiverId'].unique())

            # SimTime
            t_min = float(chunk['simTime'].min())
            t_max = float(chunk['simTime'].max())
            if t_min < time_min: time_min = t_min
            if t_max > time_max: time_max = t_max

            # Pair sequence counts
            pair_counts = chunk.groupby(['receiverId', 'senderId']).size()
            for pair, cnt in pair_counts.items():
                pair_seq_lengths[pair] += cnt

            # Subsample for separability analysis if this is train_1..train_5
            if fname in ['train_1.csv', 'train_2.csv', 'train_3.csv', 'train_4.csv', 'train_5.csv', 'train_6.csv']:
                sub = chunk.sample(n=min(len(chunk), 5000), random_state=42)
                numeric_cols = [c for c in sub.columns if c not in EXCLUDE_FROM_FEATURES and pd.api.types.is_numeric_dtype(sub[c])]
                for atk, grp in sub.groupby('attackType'):
                    for col in numeric_cols:
                        train_feat_by_attack[col][atk].extend(grp[col].dropna().tolist())

            log(f"  Processed chunk {chunk_idx + 1} ({total_rows:,} rows so far)...")

        # Identify constant columns (where min == max)
        constant_cols = [col for col in col_min if col_min[col] == col_max[col]]

        # Sequence length statistics
        seq_lens = np.array(list(pair_seq_lengths.values()))
        q25, q50, q75, q90, q95, q99 = np.percentile(seq_lens, [25, 50, 75, 90, 95, 99])

        log(f"Shape: ({total_rows:,}, {len(col_dtypes)})")
        log(f"SimTime Range: [{time_min:.4f}, {time_max:.4f}] s (Duration: {time_max - time_min:.2f} s)")
        log(f"Unique Senders: {len(unique_senders)}, Unique Receivers: {len(unique_receivers)}")
        log(f"Unique (Receiver, Sender) Pairs: {len(pair_seq_lengths):,}")
        log(f"Per-Pair Sequence Lengths: Mean={seq_lens.mean():.1f}, Std={seq_lens.std():.1f}, Min={seq_lens.min()}, Max={seq_lens.max()}")
        log(f"  Percentiles: p25={q25:.0f}, p50(Median)={q50:.0f}, p75={q75:.0f}, p90={q90:.0f}, p95={q95:.0f}, p99={q99:.0f}")
        log(f"Constant Columns ({len(constant_cols)}): {constant_cols}")
        for c in constant_cols:
            log(f"   {c} = {col_min[c]}")

        total_nans = sum(nan_counts.values())
        total_infs = sum(inf_counts.values())
        log(f"NaN Count: {total_nans}, Inf Count: {total_infs}")
        if total_nans > 0:
            log(f"  NaN Breakdown: {dict(nan_counts)}")
        if total_infs > 0:
            log(f"  Inf Breakdown: {dict(inf_counts)}")

        log(f"Class Balance (label):")
        for k in sorted(label_counts.keys()):
            pct = 100.0 * label_counts[k] / total_rows
            log(f"  Label {k}: {label_counts[k]:,} ({pct:.2f}%)")

        log(f"Attack Type Breakdown:")
        for k, v in sorted(attack_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * v / total_rows
            log(f"  {k:16s}: {v:8,} ({pct:6.2f}%)")

        file_summaries[fname] = {
            'rows': total_rows,
            'constants': constant_cols,
            'labels': dict(label_counts),
            'attacks': dict(attack_counts),
            'seq_p50': q50,
            'seq_p90': q90,
            'pairs': len(pair_seq_lengths)
        }

    # Separability and Label Proxy Analysis on Train Data
    log("\n" + "=" * 80)
    log("FEATURE SEPARABILITY & POTENTIAL LABEL PROXY ANALYSIS (TRAIN DATA)")
    log("=" * 80)

    all_attacks = sorted(list(set(atk for col in train_feat_by_attack for atk in train_feat_by_attack[col])))
    log(f"Attacks evaluated: {all_attacks}\n")

    # Header
    hdr = f"{'Feature':<20}" + "".join([f"{atk:>15}" for atk in all_attacks])
    log(hdr)
    log("-" * len(hdr))

    proxy_warnings = []

    for col in sorted(train_feat_by_attack.keys()):
        medians = {}
        for atk in all_attacks:
            vals = train_feat_by_attack[col].get(atk, [])
            medians[atk] = float(np.median(vals)) if len(vals) > 0 else np.nan

        line = f"{col:<20}" + "".join([f"{medians[atk]:15.3f}" if not np.isnan(medians[atk]) else f"{'N/A':>15}" for atk in all_attacks])
        log(line)

        # Check for potential proxies
        benign_val = medians.get('Benign', np.nan)
        for atk in all_attacks:
            if atk != 'Benign' and not np.isnan(benign_val):
                m_val = medians.get(atk, np.nan)
                # Check for extreme divergence or discrete indicator
                if (col == 'sameLocCount' or col == 'speedDelta' or col == 'posRadioMismatch') and m_val != benign_val:
                    # Physical features that indicate specific attacks
                    pass

    log("\n" + "=" * 80)
    log("SUMMARY OF CONSTANT COLUMNS ACROSS ALL 10 FILES:")
    all_const = set()
    for fname, summ in file_summaries.items():
        all_const.update(summ['constants'])
    for c in sorted(all_const):
        files_with_c = [fname for fname, s in file_summaries.items() if c in s['constants']]
        log(f"  - '{c}' is constant in {len(files_with_c)}/10 files (Files: {files_with_c})")

    log("\n" + "=" * 80)
    log("DATASET TOTAL ROWS AND CLASS DISTRIBUTION ACROSS ALL FILES:")
    total_all_rows = sum(s['rows'] for s in file_summaries.values())
    total_benign = sum(s['labels'].get(0, 0) for s in file_summaries.values())
    total_malicious = sum(s['labels'].get(1, 0) for s in file_summaries.values())
    log(f"Total Rows: {total_all_rows:,}")
    log(f"Total Benign: {total_benign:,} ({100.0 * total_benign / total_all_rows:.2f}%)")
    log(f"Total Malicious: {total_malicious:,} ({100.0 * total_malicious / total_all_rows:.2f}%)")
    log("=" * 80)

    with open(os.path.join("work", "audit_report.txt"), "w") as f:
        f.write("\n".join(output_lines))
    print(f"\nAudit report successfully saved to ./work/audit_report.txt", flush=True)

if __name__ == '__main__':
    audit()
