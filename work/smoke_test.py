import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_recall_fscore_support, average_precision_score, accuracy_score

from preprocess import RobustPreprocessor, extract_windows_from_file, CLASS_NAMES, BASE_32_FEATURES, ENGINEERED_FEATURES
from models import MultiHeadLSTM, FocalLoss

sys.stdout.reconfigure(line_buffering=True)

def run_smoke_test():
    print("=" * 80)
    print("SMOKE TEST: 1-2% DATA SUBSET (WINDOW W=20, STRIDE=5 TRAIN / 10 VAL)")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Device: {device} (torch.cuda.is_available() = {torch.cuda.is_available()})")
    
    train_files = ['train_1.csv', 'train_2.csv', 'train_3.csv', 'train_4.csv']
    val_files = ['train_5.csv', 'train_6.csv']
    
    # 1. Fit scaler on ~1.5% sample of train_1.csv
    print("\nFitting RobustPreprocessor on train sample...")
    sample_df = pd.read_csv('train_1.csv', nrows=25000)
    # Add dummy engineered features for fitting
    sample_df['dt'] = 0.1
    sample_df['posJump'] = 1.0
    sample_df['posJumpVsImplied'] = 0.1
    sample_df['speedDelta_seq'] = 0.0
    sample_df['headingDelta_seq'] = 0.0
    
    scaler = RobustPreprocessor(use_absolute_pos=False)
    scaler.fit(sample_df)
    scaler.save(os.path.join('work', 'scaler_smoke.pkl'))
    
    # 2. Extract windows from 1.5% of sequences (50 sequences per file)
    print("\nExtracting windows for smoke train (50 sequences/file)...")
    train_X_list, train_yb_list, train_ym_list = [], [], []
    for f in train_files:
        X_w, yb, ym = extract_windows_from_file(f, preprocessor=scaler, window_size=20, stride=5, max_sequences=50)
        train_X_list.append(X_w)
        train_yb_list.append(yb)
        train_ym_list.append(ym)
        
    X_train = np.concatenate(train_X_list, axis=0)
    y_train_b = np.concatenate(train_yb_list, axis=0)
    y_train_m = np.concatenate(train_ym_list, axis=0)
    print(f"Smoke Train Windows: {X_train.shape}, Label dist: {np.bincount(y_train_m, minlength=4)}")
    
    print("\nExtracting windows for smoke val (50 sequences/file)...")
    val_X_list, val_yb_list, val_ym_list = [], [], []
    for f in val_files:
        X_w, yb, ym = extract_windows_from_file(f, preprocessor=scaler, window_size=20, stride=10, max_sequences=50)
        val_X_list.append(X_w)
        val_yb_list.append(yb)
        val_ym_list.append(ym)
        
    X_val = np.concatenate(val_X_list, axis=0)
    y_val_b = np.concatenate(val_yb_list, axis=0)
    y_val_m = np.concatenate(val_ym_list, axis=0)
    print(f"Smoke Val Windows: {X_val.shape}, Label dist: {np.bincount(y_val_m, minlength=4)}")
    
    # 3. PyTorch DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_b, dtype=torch.long),
        torch.tensor(y_train_m, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val_b, dtype=torch.long),
        torch.tensor(y_val_m, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # 4. Class weights for loss
    counts_m = np.bincount(y_train_m, minlength=4)
    weights_m = len(y_train_m) / (4.0 * np.maximum(counts_m, 1))
    weights_m_tensor = torch.tensor(weights_m, dtype=torch.float32).to(device)
    
    counts_b = np.bincount(y_train_b, minlength=2)
    weights_b = len(y_train_b) / (2.0 * np.maximum(counts_b, 1))
    weights_b_tensor = torch.tensor(weights_b, dtype=torch.float32).to(device)
    
    criterion_bin = nn.CrossEntropyLoss(weight=weights_b_tensor)
    criterion_multi = nn.CrossEntropyLoss(weight=weights_m_tensor)
    
    # 5. Initialize Model
    input_size = X_train.shape[2] # 37 features
    model = MultiHeadLSTM(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.3, num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    
    print(f"\nModel initialized: MultiHeadLSTM(input_size={input_size}, hidden=128, layers=2).")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 6. Train for 3 smoke epochs
    print("\nStarting 3-epoch Smoke Training...")
    for epoch in range(1, 4):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for batch_x, batch_yb, batch_ym in train_loader:
            batch_x = batch_x.to(device)
            batch_yb = batch_yb.to(device)
            batch_ym = batch_ym.to(device)
            
            optimizer.zero_grad()
            out_bin, out_multi = model(batch_x)
            
            loss_bin = criterion_bin(out_bin, batch_yb)
            loss_multi = criterion_multi(out_multi, batch_ym)
            loss = loss_bin + loss_multi
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
            
        train_loss /= len(train_dataset)
        dt = time.time() - t0
        print(f"  Epoch {epoch}/3 - Loss: {train_loss:.4f} (Time: {dt:.2f}s)")
        
    # 7. Evaluate on Smoke Val
    model.eval()
    val_preds_b = []
    val_probs_b = []
    val_preds_m = []
    val_targets_b = []
    val_targets_m = []
    
    latencies = []
    
    with torch.no_grad():
        for batch_x, batch_yb, batch_ym in val_loader:
            batch_x = batch_x.to(device)
            
            t_start = time.perf_counter()
            out_bin, out_multi = model(batch_x)
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) / len(batch_x))
            
            probs_b = torch.softmax(out_bin, dim=1)[:, 1].cpu().numpy()
            preds_b = torch.argmax(out_bin, dim=1).cpu().numpy()
            preds_m = torch.argmax(out_multi, dim=1).cpu().numpy()
            
            val_probs_b.extend(probs_b)
            val_preds_b.extend(preds_b)
            val_preds_m.extend(preds_m)
            val_targets_b.extend(batch_yb.numpy())
            val_targets_m.extend(batch_ym.numpy())
            
    val_preds_b = np.array(val_preds_b)
    val_probs_b = np.array(val_probs_b)
    val_preds_m = np.array(val_preds_m)
    val_targets_b = np.array(val_targets_b)
    val_targets_m = np.array(val_targets_m)
    
    acc = accuracy_score(val_targets_m, val_preds_m)
    macro_f1 = f1_score(val_targets_m, val_preds_m, average='macro', zero_division=0)
    bin_f1 = f1_score(val_targets_b, val_preds_b, zero_division=0)
    pr_auc = average_precision_score(val_targets_b, val_probs_b)
    
    p, r, f1, sup = precision_recall_fscore_support(val_targets_m, val_preds_m, labels=[0, 1, 2, 3], zero_division=0)
    
    avg_latency_ms = np.mean(latencies) * 1000.0
    
    print("\n" + "=" * 80)
    print("SMOKE TEST VALIDATION RESULTS")
    print("=" * 80)
    print(f"Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f} | Binary F1: {bin_f1:.4f} | Binary PR-AUC: {pr_auc:.4f}")
    print(f"Inference Latency: {avg_latency_ms:.4f} ms per window")
    print("\nPer-Class Breakdown:")
    for i, c in enumerate(CLASS_NAMES):
        print(f"  {c:<12}: Precision={p[i]:.4f}, Recall={r[i]:.4f}, F1={f1[i]:.4f} (Support={sup[i]:,})")
    print("=" * 80)
    print("SMOKE TEST COMPLETE: Pipeline verified end-to-end successfully.")

if __name__ == '__main__':
    run_smoke_test()
