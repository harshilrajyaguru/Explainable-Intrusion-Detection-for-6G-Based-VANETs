import os
import sys
import json
import time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

CLASS_MAP = {'Benign': 0, 'Blackhole': 1, 'DDoS': 2, 'Sybil': 3}
FILES = [
    'train_1.csv', 'train_2.csv', 'train_3.csv', 'train_4.csv', 'train_5.csv', 'train_6.csv',
    'val_7.csv', 'test_8.csv', 'test_9.csv', 'test_hard_10.csv'
]

os.makedirs(os.path.join("work", "cache"), exist_ok=True)
os.makedirs(os.path.join("work", "results_v2"), exist_ok=True)

print("=" * 80)
print("EXTRACTING & CACHING V2 SEQUENCE METADATA, ONSETS, & CLUSTER IDS")
print("=" * 80)

for fname in FILES:
    base = os.path.splitext(fname)[0]
    out_meta_w20 = os.path.join("work", "cache", f"{base}_meta_W20.npz")
    out_meta_w10 = os.path.join("work", "cache", f"{base}_meta_W10.npz")
    out_meta_w40 = os.path.join("work", "cache", f"{base}_meta_W40.npz")
    out_onsets = os.path.join("work", "cache", f"{base}_attack_onsets.json")
    
    if os.path.exists(out_meta_w20) and os.path.exists(out_meta_w10) and os.path.exists(out_meta_w40) and os.path.exists(out_onsets):
        print(f"Metadata for {fname} already cached, skipping...")
        continue
        
    t0 = time.time()
    print(f"Reading {fname}...", flush=True)
    cols = ['runId', 'receiverId', 'senderId', 'simTime', 'label', 'attackType']
    df = pd.read_csv(fname, usecols=cols)
    df.sort_values(by=['runId', 'receiverId', 'senderId', 'simTime'], inplace=True)
    
    df['y_multi'] = df['attackType'].map(CLASS_MAP).values.astype(np.int64)
    df['y_bin'] = df['label'].values.astype(np.int64)
    
    # Global sequence key for cluster bootstrap: (receiverId, senderId)
    # Using receiverId_senderId as cluster identifier
    df['pair_key'] = df['receiverId'].astype(str) + '_' + df['senderId'].astype(str)
    unique_pairs = df['pair_key'].unique()
    pair_to_id = {k: i for i, k in enumerate(unique_pairs)}
    df['pair_id'] = df['pair_key'].map(pair_to_id).values.astype(np.int32)
    
    # 1. Extract attack episodes / onsets
    attack_episodes = []
    groups = df.groupby(['runId', 'receiverId', 'senderId'], sort=False)
    
    for (rId, rxId, sxId), grp in groups:
        atks = grp['attackType'].values
        times = grp['simTime'].values
        grp_len = len(atks)
        
        i = 0
        while i < grp_len:
            if atks[i] != 'Benign':
                atk_name = atks[i]
                b_start = i
                t_start = times[i]
                while i < grp_len and atks[i] == atk_name:
                    i += 1
                b_end = i - 1
                t_end = times[b_end]
                attack_episodes.append({
                    'runId': str(rId),
                    'receiverId': int(rxId),
                    'senderId': int(sxId),
                    'pair_id': int(pair_to_id[f"{rxId}_{sxId}"]),
                    'attackType': atk_name,
                    'b_onset': int(b_start),
                    't_onset': float(t_start),
                    'b_end': int(b_end),
                    't_end': float(t_end),
                    'duration_sec': float(t_end - t_start),
                    'duration_beacons': int(b_end - b_start + 1)
                })
            else:
                i += 1
                
    with open(out_onsets, 'w') as f:
        json.dump(attack_episodes, f, indent=2)
    print(f"  Saved {len(attack_episodes):,} attack episodes to {out_onsets}")
    
    # 2. Extract window metadata for W=20, W=10, W=40
    for w_size, out_path in [(20, out_meta_w20), (10, out_meta_w10), (40, out_meta_w40)]:
        stride = 10
        w_y_multi_seq = []
        w_y_bin_seq = []
        w_seq_id = []
        w_sim_time = []
        w_b_idx = []
        
        for (rId, rxId, sxId), grp in groups:
            grp_len = len(grp)
            p_id = pair_to_id[f"{rxId}_{sxId}"]
            grp_ym = grp['y_multi'].values
            grp_yb = grp['y_bin'].values
            grp_t = grp['simTime'].values
            
            if grp_len >= w_size:
                for w_start in range(0, grp_len - w_size + 1, stride):
                    w_end = w_start + w_size
                    w_y_multi_seq.append(grp_ym[w_start:w_end])
                    w_y_bin_seq.append(grp_yb[w_start:w_end])
                    w_seq_id.append(p_id)
                    w_sim_time.append(grp_t[w_end - 1])
                    w_b_idx.append(w_end - 1)
            else:
                pad_len = w_size - grp_len
                pad_ym = np.pad(grp_ym, (pad_len, 0), mode='edge')
                pad_yb = np.pad(grp_yb, (pad_len, 0), mode='edge')
                w_y_multi_seq.append(pad_ym)
                w_y_bin_seq.append(pad_yb)
                w_seq_id.append(p_id)
                w_sim_time.append(grp_t[-1])
                w_b_idx.append(grp_len - 1)
                
        np.savez_compressed(
            out_path,
            y_multi_seq=np.array(w_y_multi_seq, dtype=np.int64),
            y_bin_seq=np.array(w_y_bin_seq, dtype=np.int64),
            seq_id=np.array(w_seq_id, dtype=np.int32),
            sim_time=np.array(w_sim_time, dtype=np.float64),
            b_idx=np.array(w_b_idx, dtype=np.int32)
        )
        print(f"  Cached {len(w_seq_id):,} windows (W={w_size}) to {out_path}")
        
    print(f"Completed {fname} in {time.time()-t0:.2f}s\n")

print("All metadata successfully generated and cached!")
