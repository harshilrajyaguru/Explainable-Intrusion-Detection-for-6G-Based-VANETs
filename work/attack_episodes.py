import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

FILES = ['train_1.csv', 'train_2.csv', 'train_3.csv', 'train_4.csv', 'train_5.csv', 'train_6.csv']

print("=" * 80)
print("ANALYZING ATTACK EPISODE DURATIONS (TRAINING DATA: train_1..6)")
print("=" * 80)

episode_beacons = defaultdict(list)
episode_durations = defaultdict(list)

for fname in FILES:
    print(f"Scanning episodes in {fname}...", flush=True)
    cols = ['runId', 'receiverId', 'senderId', 'simTime', 'attackType']
    df = pd.read_csv(fname, usecols=cols)
    df.sort_values(by=['runId', 'receiverId', 'senderId', 'simTime'], inplace=True)
    
    # Group by (runId, receiverId, senderId)
    for _, grp in df.groupby(['runId', 'receiverId', 'senderId'], sort=False):
        atks = grp['attackType'].values
        times = grp['simTime'].values
        
        if len(atks) == 0:
            continue
            
        cur_atk = atks[0]
        start_t = times[0]
        b_count = 1
        
        for i in range(1, len(atks)):
            if atks[i] == cur_atk:
                b_count += 1
            else:
                dur = times[i-1] - start_t
                episode_beacons[cur_atk].append(b_count)
                episode_durations[cur_atk].append(dur)
                cur_atk = atks[i]
                start_t = times[i]
                b_count = 1
                
        # last episode in group
        dur = times[-1] - start_t
        episode_beacons[cur_atk].append(b_count)
        episode_durations[cur_atk].append(dur)

print("\n" + "=" * 80)
print("EPISODE DURATION SUMMARY PER ATTACK TYPE")
print("=" * 80)

for atk in sorted(episode_beacons.keys()):
    b_arr = np.array(episode_beacons[atk])
    d_arr = np.array(episode_durations[atk])
    
    print(f"\nAttack Type: {atk} (Total Episodes: {len(b_arr):,})")
    print(f"  Duration in Beacons:")
    print(f"    Mean: {b_arr.mean():.1f}, Std: {b_arr.std():.1f}, Min: {b_arr.min()}, Max: {b_arr.max()}")
    print(f"    p25: {np.percentile(b_arr, 25):.0f}, Median: {np.median(b_arr):.0f}, p75: {np.percentile(b_arr, 75):.0f}, p90: {np.percentile(b_arr, 90):.0f}, p95: {np.percentile(b_arr, 95):.0f}")
    print(f"  Duration in Seconds:")
    print(f"    Mean: {d_arr.mean():.2f}s, Std: {d_arr.std():.2f}s, Min: {d_arr.min():.4f}s, Max: {d_arr.max():.2f}s")
    print(f"    p25: {np.percentile(d_arr, 25):.2f}s, Median: {np.median(d_arr):.2f}s, p75: {np.percentile(d_arr, 75):.2f}s, p90: {np.percentile(d_arr, 90):.2f}s, p95: {np.percentile(d_arr, 95):.2f}s")

print("=" * 80)
