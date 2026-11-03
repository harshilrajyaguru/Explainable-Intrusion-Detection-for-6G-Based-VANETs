import os
import sys
import time
import json
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

from models import MultiHeadLSTM, ManyToManyLSTM
from evaluate import tune_multiclass_thresholds, compute_all_metrics
from evaluate_v2 import (
    compute_cluster_bootstrap_ci,
    compute_time_to_detect,
    compute_false_alarm_rate,
    CLASS_NAMES
)

sys.stdout.reconfigure(line_buffering=True)
torch.set_num_threads(16)

TRAIN_FILES = ['train_1.csv', 'train_2.csv', 'train_3.csv', 'train_4.csv']
VAL_FILES = ['train_5.csv', 'train_6.csv']
EVAL_SETS = {
    'val': VAL_FILES,
    'val_7_llm': ['val_7.csv'],
    'test_8': ['test_8.csv'],
    'test_9': ['test_9.csv'],
    'test_hard_10': ['test_hard_10.csv']
}

SEEDS = [42, 43, 44, 45, 46]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(os.path.join("work", "results_v2"), exist_ok=True)
os.makedirs(os.path.join("work", "checkpoints_v2"), exist_ok=True)

def load_cached_dataset_v2(file_list, w_size=20, compute_rolling=False):
    X_seq_list, y_bin_list, y_multi_list, is_pure_list, X_roll_list = [], [], [], [], []
    y_m_seq_list, y_b_seq_list, seq_id_list, sim_t_list, b_idx_list = [], [], [], [], []
    all_onsets = []
    
    seq_id_offset = 0
    for f in file_list:
        base = os.path.splitext(f)[0]
        d_win = np.load(os.path.join("work", "cache", f"{base}_W{w_size}_s10_no_pos.npz"))
        X_seq_list.append(d_win['X_seq'])
        y_bin_list.append(d_win['y_bin'])
        y_multi_list.append(d_win['y_multi'])
        is_pure_list.append(d_win['is_pure'])
        if compute_rolling and 'X_rolling' in d_win.files and d_win['X_rolling'] is not None:
            X_roll_list.append(d_win['X_rolling'])
            
        d_meta = np.load(os.path.join("work", "cache", f"{base}_meta_W{w_size}.npz"))
        if 'y_multi_seq' in d_meta.files:
            y_m_seq_list.append(d_meta['y_multi_seq'])
            y_b_seq_list.append(d_meta['y_bin_seq'])
        seq_id_list.append(d_meta['seq_id'] + seq_id_offset)
        sim_t_list.append(d_meta['sim_time'])
        b_idx_list.append(d_meta['b_idx'])
        
        onsets_path = os.path.join("work", "cache", f"{base}_attack_onsets.json")
        if os.path.exists(onsets_path):
            with open(onsets_path, 'r') as f_on:
                onsets = json.load(f_on)
                for on in onsets:
                    on_copy = on.copy()
                    on_copy['pair_id'] += seq_id_offset
                    all_onsets.append(on_copy)
                    
        seq_id_offset += int(np.max(d_meta['seq_id']) + 1)
        
    return {
        'X_seq': np.concatenate(X_seq_list, axis=0),
        'y_bin': np.concatenate(y_bin_list, axis=0),
        'y_multi': np.concatenate(y_multi_list, axis=0),
        'is_pure': np.concatenate(is_pure_list, axis=0),
        'X_rolling': np.concatenate(X_roll_list, axis=0) if len(X_roll_list) > 0 else None,
        'y_multi_seq': np.concatenate(y_m_seq_list, axis=0) if len(y_m_seq_list) > 0 else None,
        'y_bin_seq': np.concatenate(y_b_seq_list, axis=0) if len(y_b_seq_list) > 0 else None,
        'seq_id': np.concatenate(seq_id_list, axis=0),
        'sim_time': np.concatenate(sim_t_list, axis=0),
        'b_idx': np.concatenate(b_idx_list, axis=0),
        'attack_onsets': all_onsets
    }

def compute_sqrt_inv_weights(y_multi, y_bin):
    counts_m = np.bincount(y_multi, minlength=4)
    total_m = len(y_multi)
    w_m = np.sqrt(total_m / (4.0 * np.maximum(counts_m, 1)))
    
    counts_b = np.bincount(y_bin, minlength=2)
    total_b = len(y_bin)
    w_b = np.sqrt(total_b / (2.0 * np.maximum(counts_b, 1)))
    return torch.tensor(w_m, dtype=torch.float32).to(DEVICE), torch.tensor(w_b, dtype=torch.float32).to(DEVICE), w_m

def evaluate_predictions_v2(ym_true, yb_true, probs_m, probs_b, d_data, multipliers=None):
    metrics, preds_m = compute_all_metrics(ym_true, yb_true, probs_m, probs_b, is_pure=d_data['is_pure'], multipliers=multipliers)
    
    # 1. Cluster Bootstrap 95% CI
    ci_res = compute_cluster_bootstrap_ci(ym_true, preds_m, d_data['seq_id'], n_boot=500)
    metrics.update(ci_res)
    
    # 2. Time-to-Detect (TTD)
    ttd_res = compute_time_to_detect(d_data['attack_onsets'], preds_m, d_data['sim_time'], d_data['b_idx'], d_data['seq_id'])
    metrics['time_to_detect'] = ttd_res
    
    # 3. False-Alarm Rate (FAR)
    far_res = compute_false_alarm_rate(ym_true, preds_m, d_data['sim_time'], d_data['seq_id'])
    metrics['false_alarm_rate'] = far_res
    
    return metrics

def train_windowed_lightgbm_seed(seed, train_data, val_data, eval_datasets):
    print(f"\n--- [LightGBM Windowed] Running Seed {seed} ---", flush=True)
    X_tr = train_data['X_rolling']
    ym_tr = train_data['y_multi']
    X_va = val_data['X_rolling']
    ym_va = val_data['y_multi']
    
    counts_m = np.bincount(ym_tr, minlength=4)
    w_m = np.sqrt(len(ym_tr) / (4.0 * np.maximum(counts_m, 1)))
    sample_weights = w_m[ym_tr]
    
    clf = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=4,
        n_estimators=150,
        learning_rate=0.1,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=seed
    )
    clf.fit(X_tr, ym_tr, sample_weight=sample_weights,
            eval_set=[(X_va, ym_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)])
            
    val_probs_m = clf.predict_proba(X_va)
    multipliers = tune_multiclass_thresholds(ym_va, val_probs_m)
    
    res_seed = {}
    for dname, d_data in eval_datasets.items():
        x_eval = d_data['X_rolling']
        yb_eval = d_data['y_bin']
        ym_eval = d_data['y_multi']
        probs_m = clf.predict_proba(x_eval)
        probs_b = np.column_stack([probs_m[:, 0], 1.0 - probs_m[:, 0]])
        
        metrics = evaluate_predictions_v2(ym_eval, yb_eval, probs_m, probs_b, d_data, multipliers=multipliers)
        res_seed[dname] = metrics
        print(f"  {dname:<12} | Macro-F1: {metrics['macro_f1']:.4f} (95% CI: [{metrics['macro_f1_ci95'][0]:.4f}, {metrics['macro_f1_ci95'][1]:.4f}]) | Blackhole F1: {metrics['blackhole']['f1']:.4f}")
    return res_seed

def train_lstm_seed(model_name, seed, train_data, val_data, eval_datasets, input_size=37, num_epochs=5, patience=3):
    print(f"\n--- [{model_name}] Running Seed {seed} ---", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X_tr = train_data['X_seq']
    yb_tr = train_data['y_bin']
    ym_tr = train_data['y_multi']
    
    X_va = val_data['X_seq']
    yb_va = val_data['y_bin']
    ym_va = val_data['y_multi']
    
    w_m_tensor, w_b_tensor, _ = compute_sqrt_inv_weights(ym_tr, yb_tr)
    criterion_multi = nn.CrossEntropyLoss(weight=w_m_tensor)
    criterion_bin = nn.CrossEntropyLoss(weight=w_b_tensor)
    
    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                             torch.tensor(yb_tr, dtype=torch.long),
                             torch.tensor(ym_tr, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_va, dtype=torch.float32),
                           torch.tensor(yb_va, dtype=torch.long),
                           torch.tensor(ym_va, dtype=torch.long))
                           
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, pin_memory=False)
    
    model = MultiHeadLSTM(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.3, num_classes=4).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    
    best_macro_f1 = -1.0
    best_weights = None
    epochs_no_improve = 0
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for bx, byb, bym in train_loader:
            bx, byb, bym = bx.to(DEVICE), byb.to(DEVICE), bym.to(DEVICE)
            optimizer.zero_grad()
            out_b, out_m = model(bx)
            loss = criterion_bin(out_b, byb) + criterion_multi(out_m, bym)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * len(bx)
            
        # Validation
        model.eval()
        val_probs_m = []
        with torch.no_grad():
            for bx, byb, bym in val_loader:
                bx = bx.to(DEVICE)
                _, out_m = model(bx)
                val_probs_m.append(torch.softmax(out_m, dim=1).cpu().numpy())
        val_probs_m = np.concatenate(val_probs_m, axis=0)
        raw_preds_m = np.argmax(val_probs_m, axis=1)
        val_f1 = f1_score(ym_va, raw_preds_m, average='macro', zero_division=0)
        
        print(f"  Epoch {epoch}/{num_epochs} | Loss: {running_loss/len(train_ds):.4f} | Val F1: {val_f1:.4f} | Time: {time.time()-t0:.1f}s", flush=True)
        scheduler.step(val_f1)
        
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
                
    model.load_state_dict(best_weights)
    model.eval()
    
    val_probs_m, val_probs_b = [], []
    with torch.no_grad():
        for bx, byb, bym in val_loader:
            bx = bx.to(DEVICE)
            out_b, out_m = model(bx)
            val_probs_b.append(torch.softmax(out_b, dim=1).cpu().numpy())
            val_probs_m.append(torch.softmax(out_m, dim=1).cpu().numpy())
    val_probs_m = np.concatenate(val_probs_m, axis=0)
    val_probs_b = np.concatenate(val_probs_b, axis=0)
    multipliers = tune_multiclass_thresholds(ym_va, val_probs_m)
    
    res_seed = {}
    for dname, d_data in eval_datasets.items():
        x_eval = d_data['X_seq']
        yb_eval = d_data['y_bin']
        ym_eval = d_data['y_multi']
        
        eval_ds = TensorDataset(torch.tensor(x_eval, dtype=torch.float32))
        eval_loader = DataLoader(eval_ds, batch_size=1024, shuffle=False)
        p_m, p_b = [], []
        with torch.no_grad():
            for (bx,) in eval_loader:
                bx = bx.to(DEVICE)
                out_b, out_m = model(bx)
                p_b.append(torch.softmax(out_b, dim=1).cpu().numpy())
                p_m.append(torch.softmax(out_m, dim=1).cpu().numpy())
        p_m = np.concatenate(p_m, axis=0)
        p_b = np.concatenate(p_b, axis=0)
        
        metrics = evaluate_predictions_v2(ym_eval, yb_eval, p_m, p_b, d_data, multipliers=multipliers)
        res_seed[dname] = metrics
        print(f"  {dname:<12} | Macro-F1: {metrics['macro_f1']:.4f} (95% CI: [{metrics['macro_f1_ci95'][0]:.4f}, {metrics['macro_f1_ci95'][1]:.4f}]) | Blackhole F1: {metrics['blackhole']['f1']:.4f}")
    return res_seed

def train_many_to_many_lstm_seed(seed, train_data, val_data, eval_datasets, input_size=37, num_epochs=5, patience=3):
    print(f"\n--- [LSTM_ManyToMany W=20] Running Seed {seed} ---", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X_tr = train_data['X_seq']
    ym_seq_tr = train_data['y_multi_seq']
    yb_seq_tr = train_data['y_bin_seq']
    
    X_va = val_data['X_seq']
    ym_va = val_data['y_multi']
    yb_va = val_data['y_bin']
    
    w_m_tensor, w_b_tensor, _ = compute_sqrt_inv_weights(ym_seq_tr.reshape(-1), yb_seq_tr.reshape(-1))
    criterion_multi = nn.CrossEntropyLoss(weight=w_m_tensor)
    criterion_bin = nn.CrossEntropyLoss(weight=w_b_tensor)
    
    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                             torch.tensor(yb_seq_tr, dtype=torch.long),
                             torch.tensor(ym_seq_tr, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_va, dtype=torch.float32),
                           torch.tensor(yb_va, dtype=torch.long),
                           torch.tensor(ym_va, dtype=torch.long))
                           
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, pin_memory=False)
    
    model = ManyToManyLSTM(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.3, num_classes=4).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    
    best_macro_f1 = -1.0
    best_weights = None
    epochs_no_improve = 0
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for bx, byb_seq, bym_seq in train_loader:
            bx, byb_seq, bym_seq = bx.to(DEVICE), byb_seq.to(DEVICE), bym_seq.to(DEVICE)
            optimizer.zero_grad()
            out_b, out_m = model(bx) # (B, W, 2), (B, W, 4)
            
            # Loss computed across all timesteps
            loss = criterion_bin(out_b.view(-1, 2), byb_seq.view(-1)) + criterion_multi(out_m.view(-1, 4), bym_seq.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * len(bx)
            
        # Validation on last step representation
        model.eval()
        val_probs_m = []
        with torch.no_grad():
            for bx, byb, bym in val_loader:
                bx = bx.to(DEVICE)
                _, out_m = model(bx)
                # Last step
                val_probs_m.append(torch.softmax(out_m[:, -1, :], dim=1).cpu().numpy())
        val_probs_m = np.concatenate(val_probs_m, axis=0)
        raw_preds_m = np.argmax(val_probs_m, axis=1)
        val_f1 = f1_score(ym_va, raw_preds_m, average='macro', zero_division=0)
        
        print(f"  Epoch {epoch}/{num_epochs} | Loss (all steps): {running_loss/len(train_ds):.4f} | Val F1 (last step): {val_f1:.4f} | Time: {time.time()-t0:.1f}s", flush=True)
        scheduler.step(val_f1)
        
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
                
    model.load_state_dict(best_weights)
    model.eval()
    
    val_probs_m, val_probs_b = [], []
    with torch.no_grad():
        for bx, byb, bym in val_loader:
            bx = bx.to(DEVICE)
            out_b, out_m = model(bx)
            val_probs_b.append(torch.softmax(out_b[:, -1, :], dim=1).cpu().numpy())
            val_probs_m.append(torch.softmax(out_m[:, -1, :], dim=1).cpu().numpy())
    val_probs_m = np.concatenate(val_probs_m, axis=0)
    val_probs_b = np.concatenate(val_probs_b, axis=0)
    multipliers = tune_multiclass_thresholds(ym_va, val_probs_m)
    
    res_seed = {}
    for dname, d_data in eval_datasets.items():
        x_eval = d_data['X_seq']
        yb_eval = d_data['y_bin']
        ym_eval = d_data['y_multi']
        
        eval_ds = TensorDataset(torch.tensor(x_eval, dtype=torch.float32))
        eval_loader = DataLoader(eval_ds, batch_size=1024, shuffle=False)
        p_m, p_b = [], []
        with torch.no_grad():
            for (bx,) in eval_loader:
                bx = bx.to(DEVICE)
                out_b, out_m = model(bx)
                p_b.append(torch.softmax(out_b[:, -1, :], dim=1).cpu().numpy())
                p_m.append(torch.softmax(out_m[:, -1, :], dim=1).cpu().numpy())
        p_m = np.concatenate(p_m, axis=0)
        p_b = np.concatenate(p_b, axis=0)
        
        metrics = evaluate_predictions_v2(ym_eval, yb_eval, p_m, p_b, d_data, multipliers=multipliers)
        res_seed[dname] = metrics
        print(f"  {dname:<12} | Macro-F1: {metrics['macro_f1']:.4f} (95% CI: [{metrics['macro_f1_ci95'][0]:.4f}, {metrics['macro_f1_ci95'][1]:.4f}]) | Blackhole F1: {metrics['blackhole']['f1']:.4f}")
    return res_seed

def generate_comparison_v2_markdown(all_data):
    md_path = os.path.join("work", "results_v2", "comparison_v2.md")
    models = list(all_data.keys())
    dsets = ['val', 'val_7_llm', 'test_8', 'test_9', 'test_hard_10']
    dset_labels = {
        'val': 'Val (train_5+6)',
        'val_7_llm': 'val_7 (LLM)',
        'test_8': 'test_8',
        'test_9': 'test_9',
        'test_hard_10': 'test_hard_10'
    }
    
    lines = []
    lines.append("# VANET Intrusion Detection Benchmark (V2): Multi-Seed, Bootstrap CIs & Latency Analysis\n")
    
    # 1. Macro-F1 Table: Mean ± Std and 95% Bootstrap CI
    lines.append("## 1. Macro-F1 Performance (Mean ± Std over 5 Seeds & Bootstrap 95% CI)\n")
    lines.append("| Model | " + " | ".join([dset_labels[d] for d in dsets]) + " | Mean Macro-F1 |")
    lines.append("| :--- | " + " | ".join([":---:" for _ in dsets]) + " | :---: |")
    
    for m in models:
        m_str_list = []
        mean_across_ds = []
        for d in dsets:
            # Collect scores across seeds
            f1s = [all_data[m][str(s)][d]['macro_f1'] for s in SEEDS if str(s) in all_data[m]]
            if len(f1s) > 0:
                mean_f1 = np.mean(f1s)
                std_f1 = np.std(f1s)
                mean_across_ds.append(mean_f1)
                
                # Representative bootstrap CI (from seed 42)
                ci = all_data[m][str(SEEDS[0])][d]['macro_f1_ci95']
                m_str_list.append(f"{mean_f1:.4f}±{std_f1:.4f}<br>[{ci[0]:.4f}, {ci[1]:.4f}]")
            else:
                m_str_list.append("N/A")
        mean_all = f"**{np.mean(mean_across_ds):.4f}**" if len(mean_across_ds) > 0 else "N/A"
        lines.append(f"| **{m}** | " + " | ".join(m_str_list) + f" | {mean_all} |")
        
    # 2. Blackhole F1 Table
    lines.append("\n## 2. Blackhole Attack F1-Score (Mean ± Std & 95% CI)\n")
    lines.append("| Model | " + " | ".join([dset_labels[d] for d in dsets]) + " | Mean Blackhole F1 |")
    lines.append("| :--- | " + " | ".join([":---:" for _ in dsets]) + " | :---: |")
    
    for m in models:
        m_str_list = []
        mean_across_ds = []
        for d in dsets:
            f1s = [all_data[m][str(s)][d]['blackhole']['f1'] for s in SEEDS if str(s) in all_data[m]]
            if len(f1s) > 0:
                mean_f1 = np.mean(f1s)
                std_f1 = np.std(f1s)
                mean_across_ds.append(mean_f1)
                ci = all_data[m][str(SEEDS[0])][d]['blackhole_f1_ci95']
                m_str_list.append(f"{mean_f1:.4f}±{std_f1:.4f}<br>[{ci[0]:.4f}, {ci[1]:.4f}]")
            else:
                m_str_list.append("N/A")
        mean_all = f"**{np.mean(mean_across_ds):.4f}**" if len(mean_across_ds) > 0 else "N/A"
        lines.append(f"| **{m}** | " + " | ".join(m_str_list) + f" | {mean_all} |")
        
    # 3. Statistical Significance Check (CI Separation)
    lines.append("\n## 3. Statistical Significance Analysis (Bootstrap CI Separation)\n")
    lines.append("Rule: Differences are designated **Statistically Significant** if and only if their 95% Confidence Intervals are disjoint (separate).\n")
    lines.append("| Comparison | Dataset | Model A (95% CI) | Model B (95% CI) | CIs Separate? | Significant? |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
    
    comp_pairs = [
        ('LightGBM_Windowed', 'LSTM_W10_NoPos'),
        ('LightGBM_Windowed', 'LSTM_W40_NoPos'),
        ('LSTM_W10_NoPos', 'LSTM_ManyToMany'),
        ('LSTM_W20_NoPos', 'LSTM_ManyToMany') if 'LSTM_W20_NoPos' in all_data else ('LightGBM_Windowed', 'LSTM_ManyToMany')
    ]
    
    for m_a, m_b in comp_pairs:
        if m_a in all_data and m_b in all_data and str(SEEDS[0]) in all_data[m_a] and str(SEEDS[0]) in all_data[m_b]:
            for d in ['val', 'test_hard_10']:
                ci_a = all_data[m_a][str(SEEDS[0])][d]['macro_f1_ci95']
                ci_b = all_data[m_b][str(SEEDS[0])][d]['macro_f1_ci95']
                # Check separation
                separate = (ci_a[1] < ci_b[0]) or (ci_b[1] < ci_a[0])
                sig_str = "**YES** (Significant)" if separate else "NO (Overlapping)"
                lines.append(f"| {m_a} vs {m_b} | {dset_labels[d]} | [{ci_a[0]:.4f}, {ci_a[1]:.4f}] | [{ci_b[0]:.4f}, {ci_b[1]:.4f}] | {'YES' if separate else 'NO'} | {sig_str} |")

    # 4. Time-to-Detect (TTD) Table
    lines.append("\n## 4. Time-to-Detect (TTD) After Attack Onset\n")
    lines.append("Evaluated on unseen test scenarios (`test_8` and `test_hard_10`). Reported as Median / 90th percentile (seconds & beacons).\n")
    lines.append("| Model | Attack Type | Detection Rate | TTD Median (s) | TTD p90 (s) | TTD Median (beacons) | TTD p90 (beacons) |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    
    for m in models:
        if str(SEEDS[0]) in all_data[m]:
            ttd_data = all_data[m][str(SEEDS[0])]['test_hard_10'].get('time_to_detect', {})
            for atk in ['Blackhole', 'DDoS', 'Sybil']:
                if atk in ttd_data:
                    info = ttd_data[atk]
                    lines.append(f"| {m} | **{atk}** | {info['detection_rate']*100:.1f}% | {info['median_seconds']:.2f}s | {info['p90_seconds']:.2f}s | {info['median_beacons']:.0f} | {info['p90_beacons']:.0f} |")

    # 5. False-Alarm Rate (FAR) Table
    lines.append("\n## 5. False-Alarm Rate (FAR) on Benign Traffic\n")
    lines.append("| Model | Val (alarms/min) | val_7 (alarms/min) | test_8 (alarms/min) | test_9 (alarms/min) | test_hard_10 (alarms/min) | Mean FAR |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for m in models:
        far_vals = []
        far_str_list = []
        for d in dsets:
            if str(SEEDS[0]) in all_data[m]:
                far_val = all_data[m][str(SEEDS[0])][d].get('false_alarm_rate', {}).get('false_alarm_rate_per_min', np.nan)
                far_vals.append(far_val)
                far_str_list.append(f"{far_val:.2f}")
            else:
                far_str_list.append("N/A")
        mean_far = f"**{np.nanmean(far_vals):.2f}**" if len(far_vals) > 0 else "N/A"
        lines.append(f"| **{m}** | " + " | ".join(far_str_list) + f" | {mean_far} |")
        
    with open(md_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"[Report] Updated comparison_v2.md at {md_path}", flush=True)

def run_v2_pipeline():
    print("=" * 80)
    print("STARTING V2 EXPERIMENTS: 5 SEEDS, BOOTSTRAP CIs, TTD & MANY-TO-MANY LSTM")
    print("=" * 80)
    
    v2_json_path = os.path.join("work", "results_v2", "metrics_v2.json")
    if os.path.exists(v2_json_path):
        try:
            with open(v2_json_path, 'r') as f:
                all_results = json.load(f)
            print(f"Loaded existing V2 metrics: {list(all_results.keys())}")
        except Exception:
            all_results = {}
    else:
        all_results = {}
        
    def checkpoint():
        with open(v2_json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        generate_comparison_v2_markdown(all_results)
        
    # 1. Load Datasets
    print("\n>>> Loading W=20 Datasets (with rolling features)...")
    train_w20 = load_cached_dataset_v2(TRAIN_FILES, w_size=20, compute_rolling=True)
    val_w20 = load_cached_dataset_v2(VAL_FILES, w_size=20, compute_rolling=True)
    eval_w20 = {d: load_cached_dataset_v2(flist, w_size=20, compute_rolling=True) for d, flist in EVAL_SETS.items()}
    
    # -------------------------------------------------------------------------
    # MODEL 1: LightGBM Windowed (5 seeds)
    # -------------------------------------------------------------------------
    if 'LightGBM_Windowed' not in all_results:
        all_results['LightGBM_Windowed'] = {}
    for s in SEEDS:
        if str(s) not in all_results['LightGBM_Windowed']:
            res = train_windowed_lightgbm_seed(s, train_w20, val_w20, eval_w20)
            all_results['LightGBM_Windowed'][str(s)] = res
            checkpoint()
        else:
            print(f"Skipping LightGBM_Windowed Seed {s} (already completed)")
            
    # -------------------------------------------------------------------------
    # MODEL 2: LSTM Many-to-Many W=20 (5 seeds)
    # -------------------------------------------------------------------------
    if 'LSTM_ManyToMany' not in all_results:
        all_results['LSTM_ManyToMany'] = {}
    for s in SEEDS:
        if str(s) not in all_results['LSTM_ManyToMany']:
            res = train_many_to_many_lstm_seed(s, train_w20, val_w20, eval_w20, input_size=37, num_epochs=5, patience=3)
            all_results['LSTM_ManyToMany'][str(s)] = res
            checkpoint()
        else:
            print(f"Skipping LSTM_ManyToMany Seed {s} (already completed)")

    # -------------------------------------------------------------------------
    # MODEL 3: LSTM W=10 (5 seeds)
    # -------------------------------------------------------------------------
    print("\n>>> Loading W=10 Datasets...")
    train_w10 = load_cached_dataset_v2(TRAIN_FILES, w_size=10, compute_rolling=False)
    val_w10 = load_cached_dataset_v2(VAL_FILES, w_size=10, compute_rolling=False)
    eval_w10 = {d: load_cached_dataset_v2(flist, w_size=10, compute_rolling=False) for d, flist in EVAL_SETS.items()}
    
    if 'LSTM_W10_NoPos' not in all_results:
        all_results['LSTM_W10_NoPos'] = {}
    for s in SEEDS:
        if str(s) not in all_results['LSTM_W10_NoPos']:
            res = train_lstm_seed("LSTM_W10_NoPos", s, train_w10, val_w10, eval_w10, input_size=37, num_epochs=5, patience=3)
            all_results['LSTM_W10_NoPos'][str(s)] = res
            checkpoint()
        else:
            print(f"Skipping LSTM_W10_NoPos Seed {s} (already completed)")

    # -------------------------------------------------------------------------
    # MODEL 4: LSTM W=40 (5 seeds)
    # -------------------------------------------------------------------------
    print("\n>>> Loading W=40 Datasets...")
    train_w40 = load_cached_dataset_v2(TRAIN_FILES, w_size=40, compute_rolling=False)
    val_w40 = load_cached_dataset_v2(VAL_FILES, w_size=40, compute_rolling=False)
    eval_w40 = {d: load_cached_dataset_v2(flist, w_size=40, compute_rolling=False) for d, flist in EVAL_SETS.items()}
    
    if 'LSTM_W40_NoPos' not in all_results:
        all_results['LSTM_W40_NoPos'] = {}
    for s in SEEDS:
        if str(s) not in all_results['LSTM_W40_NoPos']:
            res = train_lstm_seed("LSTM_W40_NoPos", s, train_w40, val_w40, eval_w40, input_size=37, num_epochs=5, patience=3)
            all_results['LSTM_W40_NoPos'][str(s)] = res
            checkpoint()
        else:
            print(f"Skipping LSTM_W40_NoPos Seed {s} (already completed)")

    print("\n================================================================================")
    print("V2 BENCHMARK COMPLETED SUCCESSFULLY! All results saved to work/results_v2/")
    print("================================================================================")

if __name__ == '__main__':
    run_v2_pipeline()
