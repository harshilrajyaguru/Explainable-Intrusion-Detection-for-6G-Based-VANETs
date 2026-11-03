import os
import sys
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score, average_precision_score

sys.stdout.reconfigure(line_buffering=True)
CLASS_NAMES = ['Benign', 'Blackhole', 'DDoS', 'Sybil']

def compute_cluster_bootstrap_ci(y_true, y_pred, cluster_ids, n_boot=500, seed=42):
    """
    Computes a cluster bootstrap 95% Confidence Interval for Macro-F1 and Blackhole F1
    by resampling (receiverId, senderId) sequences with replacement.
    """
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(cluster_ids)
    K = len(unique_clusters)
    
    # Map cluster_ids to 0..K-1
    cid_map = {c: i for i, c in enumerate(unique_clusters)}
    c_indices = np.array([cid_map[c] for c in cluster_ids], dtype=np.int32)
    
    # Pre-index windows by cluster
    cluster_to_idx = [[] for _ in range(K)]
    for idx, c in enumerate(c_indices):
        cluster_to_idx[c].append(idx)
    cluster_to_idx = [np.array(x, dtype=np.int32) for x in cluster_to_idx if len(x) > 0]
    K_eff = len(cluster_to_idx)
    
    macro_f1_samples = []
    bh_f1_samples = []
    
    for _ in range(n_boot):
        sampled_c = rng.integers(0, K_eff, size=K_eff)
        sample_idx = np.concatenate([cluster_to_idx[c] for c in sampled_c])
        
        yt = y_true[sample_idx]
        yp = y_pred[sample_idx]
        
        cm = np.bincount(4 * yt + yp, minlength=16).reshape(4, 4)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        denom = 2 * tp + fp + fn
        f1 = np.where(denom > 0, 2 * tp / denom, 0.0)
        
        macro_f1_samples.append(float(np.mean(f1)))
        bh_f1_samples.append(float(f1[1]))
        
    ci_macro = [float(np.percentile(macro_f1_samples, 2.5)), float(np.percentile(macro_f1_samples, 97.5))]
    ci_bh = [float(np.percentile(bh_f1_samples, 2.5)), float(np.percentile(bh_f1_samples, 97.5))]
    
    return {
        'macro_f1_ci95': ci_macro,
        'blackhole_f1_ci95': ci_bh
    }

def compute_time_to_detect(attack_episodes, preds, sim_times, b_indices, pair_ids):
    """
    Computes Time-to-Detect (TTD) after attack onset per attackType:
    Median and p90 in seconds and in beacons.
    """
    # Group window predictions by pair_id
    pair_to_windows = {}
    for idx, pid in enumerate(pair_ids):
        if pid not in pair_to_windows:
            pair_to_windows[pid] = []
        pair_to_windows[pid].append((b_indices[idx], sim_times[idx], preds[idx]))
        
    ttd_seconds = {'Blackhole': [], 'DDoS': [], 'Sybil': []}
    ttd_beacons = {'Blackhole': [], 'DDoS': [], 'Sybil': []}
    detected_count = {'Blackhole': 0, 'DDoS': 0, 'Sybil': 0}
    total_count = {'Blackhole': 0, 'DDoS': 0, 'Sybil': 0}
    
    for ep in attack_episodes:
        atk = ep['attackType']
        if atk not in ttd_seconds:
            continue
        total_count[atk] += 1
        pid = ep['pair_id']
        b_onset = ep['b_onset']
        t_onset = ep['t_onset']
        b_end = ep['b_end']
        t_end = ep['t_end']
        
        windows = pair_to_windows.get(pid, [])
        # Find first detection after onset
        detected = False
        for b_w, t_w, pred in windows:
            if b_w >= b_onset and t_w <= t_end + 1.0: # within or immediately upon episode
                if pred != 0: # flagged as attack
                    dt = max(0.0, t_w - t_onset)
                    db = max(1, b_w - b_onset + 1)
                    ttd_seconds[atk].append(dt)
                    ttd_beacons[atk].append(db)
                    detected = True
                    break
        if detected:
            detected_count[atk] += 1
            
    summary = {}
    for atk in ['Blackhole', 'DDoS', 'Sybil']:
        sec_arr = np.array(ttd_seconds[atk]) if len(ttd_seconds[atk]) > 0 else np.array([np.nan])
        bec_arr = np.array(ttd_beacons[atk]) if len(ttd_beacons[atk]) > 0 else np.array([np.nan])
        det_rate = (detected_count[atk] / total_count[atk]) if total_count[atk] > 0 else 0.0
        
        summary[atk] = {
            'total_episodes': total_count[atk],
            'detected_episodes': detected_count[atk],
            'detection_rate': float(det_rate),
            'median_seconds': float(np.nanmedian(sec_arr)),
            'p90_seconds': float(np.nanpercentile(sec_arr, 90)) if len(sec_arr) > 0 and not np.isnan(sec_arr[0]) else np.nan,
            'median_beacons': float(np.nanmedian(bec_arr)),
            'p90_beacons': float(np.nanpercentile(bec_arr, 90)) if len(bec_arr) > 0 and not np.isnan(bec_arr[0]) else np.nan
        }
    return summary

def compute_false_alarm_rate(y_true, y_pred, sim_times, pair_ids):
    """
    Computes False-Alarm Rate (FAR) per minute on benign traffic.
    """
    benign_mask = (y_true == 0)
    if np.sum(benign_mask) == 0:
        return {'false_alarm_rate_per_min': 0.0, 'false_alarms': 0, 'total_benign_windows': 0}
        
    false_alarms = int(np.sum(y_pred[benign_mask] != 0))
    
    # Calculate duration of benign monitoring
    # Group by pair_id to find active span
    total_sec = 0.0
    for pid in np.unique(pair_ids):
        p_mask = (pair_ids == pid) & benign_mask
        if np.sum(p_mask) > 1:
            t_slice = sim_times[p_mask]
            total_sec += (t_slice[-1] - t_slice[0])
            
    total_minutes = max(1.0, total_sec / 60.0)
    far_per_min = false_alarms / total_minutes
    
    return {
        'false_alarm_rate_per_min': float(far_per_min),
        'false_alarms': false_alarms,
        'total_benign_windows': int(np.sum(benign_mask)),
        'total_benign_minutes': float(total_minutes)
    }
