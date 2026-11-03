import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

FILES = [
    'train_1.csv', 'train_2.csv', 'train_3.csv', 'train_4.csv', 'train_5.csv', 'train_6.csv',
    'val_7.csv', 'test_8.csv', 'test_9.csv', 'test_hard_10.csv'
]

print("=" * 80)
print("AUDIT PART 2: RUNS, SENDER LABEL DYNAMICS, AND TRAJECTORY OVERLAP")
print("=" * 80)

# 1. Distinct runId per file & 2. Sender label constancy within run
file_run_ids = {}
file_sender_stats = {}
file_trajectories = defaultdict(lambda: defaultdict(set)) # file -> senderId -> set of (round(posX, 1), round(posY, 1))

for fname in FILES:
    print(f"\nScanning {fname}...", flush=True)
    run_ids = set()
    sender_labels = defaultdict(set)
    sender_attacks = defaultdict(set)
    
    # Read relevant columns only to be very fast
    cols = ['runId', 'simTime', 'senderId', 'posX', 'posY', 'label', 'attackType']
    for chunk in pd.read_csv(fname, usecols=cols, chunksize=150_000):
        run_ids.update(chunk['runId'].unique())
        
        # Track label/attackType per sender within this file
        for sender_id, grp in chunk.groupby('senderId'):
            sender_labels[sender_id].update(grp['label'].unique())
            sender_attacks[sender_id].update(grp['attackType'].unique())
            
        # Sample trajectory points: round to 1 decimal place (~10cm resolution)
        # To avoid giant sets, take every 5th beacon
        sub = chunk.iloc[::5]
        for s_id, px, py in zip(sub['senderId'], np.round(sub['posX'], 1), np.round(sub['posY'], 1)):
            file_trajectories[fname][s_id].add((px, py))
            
    file_run_ids[fname] = run_ids
    
    # Senders with multiple labels
    switching_senders = [s for s, labs in sender_labels.items() if len(labs) > 1]
    pure_benign = [s for s, labs in sender_labels.items() if labs == {0}]
    pure_malicious = [s for s, labs in sender_labels.items() if labs == {1}]
    
    file_sender_stats[fname] = {
        'total_senders': len(sender_labels),
        'pure_benign': len(pure_benign),
        'pure_malicious': len(pure_malicious),
        'switching': len(switching_senders),
        'switching_ids': switching_senders[:10]
    }
    
    print(f"  Distinct runIds: {len(run_ids)} -> {list(run_ids)}")
    print(f"  Total senders: {len(sender_labels)}")
    print(f"  Pure Benign senders: {len(pure_benign)}")
    print(f"  Pure Malicious senders: {len(pure_malicious)}")
    print(f"  Senders that switch labels mid-run (attack starts mid-run): {len(switching_senders)}")
    if len(switching_senders) > 0:
        print(f"    Example switching senderIds: {switching_senders[:5]}")

print("\n" + "=" * 80)
print("TRAJECTORY OVERLAP ANALYSIS (ROUNDED posX, posY FOR SAME senderId)")
print("=" * 80)

# Aggregate train trajectory points per senderId (train_1..4 as train, and train_1..6)
train_1_4_trajectories = defaultdict(set)
for f in ['train_1.csv', 'train_2.csv', 'train_3.csv', 'train_4.csv']:
    for s_id, pts in file_trajectories[f].items():
        train_1_4_trajectories[s_id].update(pts)

train_1_6_trajectories = defaultdict(set)
for f in ['train_1.csv', 'train_2.csv', 'train_3.csv', 'train_4.csv', 'train_5.csv', 'train_6.csv']:
    for s_id, pts in file_trajectories[f].items():
        train_1_6_trajectories[s_id].update(pts)

eval_files = ['train_5.csv', 'train_6.csv', 'val_7.csv', 'test_8.csv', 'test_9.csv', 'test_hard_10.csv']

print(f"\nOverlap of (senderId, rounded_position) with Train (train_1..4):")
for ef in eval_files:
    total_eval_points = sum(len(pts) for pts in file_trajectories[ef].values())
    overlapping_points = 0
    overlapping_senders = 0
    common_senders = set(file_trajectories[ef].keys()).intersection(set(train_1_4_trajectories.keys()))
    
    for s_id in common_senders:
        inter = file_trajectories[ef][s_id].intersection(train_1_4_trajectories[s_id])
        overlapping_points += len(inter)
        if len(inter) > 0:
            overlapping_senders += 1
            
    pct_points = (100.0 * overlapping_points / total_eval_points) if total_eval_points > 0 else 0
    pct_senders = (100.0 * overlapping_senders / len(file_trajectories[ef])) if len(file_trajectories[ef]) > 0 else 0
    print(f"  {ef:<18}: {len(common_senders)}/{len(file_trajectories[ef])} common senders | {overlapping_points:,}/{total_eval_points:,} points overlap ({pct_points:.2f}%) | {overlapping_senders} senders share position ({pct_senders:.1f}%)")

print("\nOverlap of (senderId, rounded_position) with Train (train_1..6):")
for ef in ['val_7.csv', 'test_8.csv', 'test_9.csv', 'test_hard_10.csv']:
    total_eval_points = sum(len(pts) for pts in file_trajectories[ef].values())
    overlapping_points = 0
    overlapping_senders = 0
    common_senders = set(file_trajectories[ef].keys()).intersection(set(train_1_6_trajectories.keys()))
    
    for s_id in common_senders:
        inter = file_trajectories[ef][s_id].intersection(train_1_6_trajectories[s_id])
        overlapping_points += len(inter)
        if len(inter) > 0:
            overlapping_senders += 1
            
    pct_points = (100.0 * overlapping_points / total_eval_points) if total_eval_points > 0 else 0
    pct_senders = (100.0 * overlapping_senders / len(file_trajectories[ef])) if len(file_trajectories[ef]) > 0 else 0
    print(f"  {ef:<18}: {len(common_senders)}/{len(file_trajectories[ef])} common senders | {overlapping_points:,}/{total_eval_points:,} points overlap ({pct_points:.2f}%) | {overlapping_senders} senders share position ({pct_senders:.1f}%)")

print("=" * 80)
