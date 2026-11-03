import os
import sys
import numpy as np
import pandas as pd
import pickle

sys.stdout.reconfigure(line_buffering=True)

CLASS_MAP = {'Benign': 0, 'Blackhole': 1, 'DDoS': 2, 'Sybil': 3}
CLASS_NAMES = ['Benign', 'Blackhole', 'DDoS', 'Sybil']

BASE_32_FEATURES = [
    'speed', 'accel', 'heading', 'rxSpeed', 'rxHeading',
    'distance', 'relSpeed', 'headingDiff', 'rsrp', 'sinr', 'cqi', 'rssi', 'pathLoss',
    'delay', 'jitter', 'interArrival', 'seqGap', 'pktSize', 'pktRate', 'pdr', 'fwdRatio',
    'numNeighbors', 'neighborDensity', 'cbr', 'rxMsgRate', 'sameLocCount', 'distEst',
    'posRadioMismatch', 'impliedSpeed', 'speedDelta', 'idAge', 'txPower'
]

POS_FEATURES = ['posX', 'posY', 'rxPosX', 'rxPosY']

ENGINEERED_FEATURES = [
    'dt', 'posJump', 'posJumpVsImplied', 'speedDelta_seq', 'headingDelta_seq'
]

HEAVY_TAIL_COLS = [
    'pktRate', 'rxMsgRate', 'cbr', 'posRadioMismatch', 'interArrival',
    'delay', 'pathLoss', 'posJump', 'posJumpVsImplied'
]

def engineer_group_features(df_group):
    df_group = df_group.copy()
    sim_time = df_group['simTime'].values
    dt = np.diff(sim_time, prepend=sim_time[0] + 0.1)
    dt = np.where(dt <= 0, 0.1, dt)
    
    pos_x = df_group['posX'].values
    pos_y = df_group['posY'].values
    dx = np.diff(pos_x, prepend=pos_x[0])
    dy = np.diff(pos_y, prepend=pos_y[0])
    pos_jump = np.sqrt(dx**2 + dy**2)
    
    speed = df_group['speed'].values
    pos_implied = np.abs(pos_jump - speed * dt)
    speed_delta = np.diff(speed, prepend=speed[0])
    
    heading = df_group['heading'].values
    h_delta = np.diff(heading, prepend=heading[0])
    h_delta = ((h_delta + 180.0) % 360.0) - 180.0
    
    df_group['dt'] = dt.astype(np.float32)
    df_group['posJump'] = pos_jump.astype(np.float32)
    df_group['posJumpVsImplied'] = pos_implied.astype(np.float32)
    df_group['speedDelta_seq'] = speed_delta.astype(np.float32)
    df_group['headingDelta_seq'] = h_delta.astype(np.float32)
    return df_group

class RobustPreprocessor:
    def __init__(self, use_absolute_pos=False):
        self.use_absolute_pos = use_absolute_pos
        self.feature_cols = (POS_FEATURES + BASE_32_FEATURES if use_absolute_pos else BASE_32_FEATURES) + ENGINEERED_FEATURES
        self.medians = {}
        self.iqrs = {}
        self.heavy_tails = [c for c in HEAVY_TAIL_COLS if c in self.feature_cols]

    def fit(self, df):
        print(f"Fitting RobustPreprocessor on {len(df):,} rows ({len(self.feature_cols)} features, use_pos={self.use_absolute_pos})...")
        df_copy = df[self.feature_cols].copy()
        for c in self.heavy_tails:
            df_copy[c] = np.log1p(np.maximum(0, df_copy[c]))
            
        for c in self.feature_cols:
            vals = df_copy[c].values
            med = float(np.nanmedian(vals))
            q25, q75 = np.nanpercentile(vals, [25, 75])
            iqr = float(q75 - q25)
            if iqr <= 1e-6:
                iqr = float(np.nanstd(vals))
                if iqr <= 1e-6:
                    iqr = 1.0
            self.medians[c] = med
            self.iqrs[c] = iqr
        print("RobustPreprocessor fit complete.")

    def transform(self, df):
        df_copy = df[self.feature_cols].copy()
        for c in self.heavy_tails:
            df_copy[c] = np.log1p(np.maximum(0, df_copy[c]))
            
        X = np.zeros((len(df), len(self.feature_cols)), dtype=np.float32)
        for i, c in enumerate(self.feature_cols):
            val = (df_copy[c].values - self.medians[c]) / self.iqrs[c]
            X[:, i] = np.clip(val, -10.0, 10.0)
        return X

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'use_absolute_pos': self.use_absolute_pos,
                'feature_cols': self.feature_cols,
                'medians': self.medians,
                'iqrs': self.iqrs,
                'heavy_tails': self.heavy_tails
            }, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            d = pickle.load(f)
        obj = cls(use_absolute_pos=d['use_absolute_pos'])
        obj.feature_cols = d['feature_cols']
        obj.medians = d['medians']
        obj.iqrs = d['iqrs']
        obj.heavy_tails = d['heavy_tails']
        return obj

def compute_rolling_stats(window_arr):
    # window_arr: (W, D)
    # returns: (5 * D,) -> [mean, std, min, max, last]
    w_mean = np.mean(window_arr, axis=0)
    w_std = np.std(window_arr, axis=0)
    w_min = np.min(window_arr, axis=0)
    w_max = np.max(window_arr, axis=0)
    w_last = window_arr[-1]
    return np.concatenate([w_mean, w_std, w_min, w_max, w_last])

def process_file_windows(csv_file, preprocessor, window_size=20, stride=10, compute_rolling=True, cache_dir='work/cache'):
    """
    Loads csv_file, groups by (runId, receiverId, senderId), generates sliding windows of size W,
    computes labels (last beacon), pure vs transition flags, and optional rolling stats.
    Caches to .npz for fast reuse.
    """
    fname = os.path.basename(csv_file).replace('.csv', '')
    use_pos_str = "with_pos" if preprocessor.use_absolute_pos else "no_pos"
    cache_path = os.path.join(cache_dir, f"{fname}_W{window_size}_s{stride}_{use_pos_str}.npz")
    
    if os.path.exists(cache_path):
        print(f"Loading cached windows from {cache_path}...")
        data = np.load(cache_path)
        return {
            'X_seq': data['X_seq'],
            'y_bin': data['y_bin'],
            'y_multi': data['y_multi'],
            'is_pure': data['is_pure'],
            'X_rolling': data['X_rolling'] if 'X_rolling' in data.files else None
        }

    print(f"Processing windows for {csv_file} (W={window_size}, stride={stride}, use_pos={preprocessor.use_absolute_pos})...", flush=True)
    cols_to_load = [
        'runId', 'simTime', 'receiverId', 'senderId', 'posX', 'posY', 'speed', 'accel', 'heading',
        'rxPosX', 'rxPosY', 'rxSpeed', 'rxHeading', 'distance', 'relSpeed', 'headingDiff', 'rsrp',
        'sinr', 'cqi', 'rssi', 'pathLoss', 'delay', 'jitter', 'interArrival', 'seqGap', 'pktSize',
        'pktRate', 'pdr', 'fwdRatio', 'numNeighbors', 'neighborDensity', 'cbr', 'rxMsgRate',
        'sameLocCount', 'distEst', 'posRadioMismatch', 'impliedSpeed', 'speedDelta', 'idAge', 'txPower',
        'label', 'attackType'
    ]
    df = pd.read_csv(csv_file, usecols=cols_to_load)
    df.sort_values(by=['runId', 'receiverId', 'senderId', 'simTime'], inplace=True)
    
    grouped_dfs = []
    groups = df.groupby(['runId', 'receiverId', 'senderId'], sort=False)
    for _, grp in groups:
        engineered = engineer_group_features(grp)
        grouped_dfs.append(engineered)
        
    full_df = pd.concat(grouped_dfs, ignore_index=True)
    X_feats = preprocessor.transform(full_df)
    y_bin_arr = full_df['label'].values.astype(np.int64)
    y_multi_arr = full_df['attackType'].map(CLASS_MAP).values.astype(np.int64)
    
    windows_X = []
    windows_rolling = []
    windows_y_bin = []
    windows_y_multi = []
    windows_is_pure = []
    
    idx_start = 0
    for grp in grouped_dfs:
        grp_len = len(grp)
        idx_end = idx_start + grp_len
        
        grp_X = X_feats[idx_start:idx_end]
        grp_y_b = y_bin_arr[idx_start:idx_end]
        grp_y_m = y_multi_arr[idx_start:idx_end]
        
        if grp_len >= window_size:
            for w_start in range(0, grp_len - window_size + 1, stride):
                w_end = w_start + window_size
                w_x = grp_X[w_start:w_end]
                w_ym = grp_y_m[w_start:w_end]
                
                windows_X.append(w_x)
                if compute_rolling:
                    windows_rolling.append(compute_rolling_stats(w_x))
                    
                windows_y_bin.append(grp_y_b[w_end - 1])
                windows_y_multi.append(grp_y_m[w_end - 1])
                # Pure window check: all labels in window equal
                windows_is_pure.append(len(np.unique(w_ym)) == 1)
        else:
            # Pad short sequences
            pad_len = window_size - grp_len
            pad_block = np.repeat(grp_X[0:1], pad_len, axis=0)
            padded_w = np.vstack([pad_block, grp_X])
            windows_X.append(padded_w)
            if compute_rolling:
                windows_rolling.append(compute_rolling_stats(padded_w))
            windows_y_bin.append(grp_y_b[-1])
            windows_y_multi.append(grp_y_m[-1])
            windows_is_pure.append(len(np.unique(grp_y_m)) == 1)
            
        idx_start = idx_end
        
    X_seq = np.array(windows_X, dtype=np.float32)
    y_bin = np.array(windows_y_bin, dtype=np.int64)
    y_multi = np.array(windows_y_multi, dtype=np.int64)
    is_pure = np.array(windows_is_pure, dtype=bool)
    X_rolling = np.array(windows_rolling, dtype=np.float32) if compute_rolling else None
    
    # Save cache
    save_dict = {
        'X_seq': X_seq,
        'y_bin': y_bin,
        'y_multi': y_multi,
        'is_pure': is_pure
    }
    if X_rolling is not None:
        save_dict['X_rolling'] = X_rolling
    np.savez_compressed(cache_path, **save_dict)
    print(f"Cached {len(X_seq):,} windows to {cache_path}")
    
    return {
        'X_seq': X_seq,
        'y_bin': y_bin,
        'y_multi': y_multi,
        'is_pure': is_pure,
        'X_rolling': X_rolling
    }
