import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, roc_auc_score, average_precision_score

sys.stdout.reconfigure(line_buffering=True)

# 36 Retained Features
RAW_36_FEATURES = [
    'posX', 'posY', 'speed', 'accel', 'heading', 'rxPosX', 'rxPosY', 'rxSpeed', 'rxHeading',
    'distance', 'relSpeed', 'headingDiff', 'rsrp', 'sinr', 'cqi', 'rssi', 'pathLoss',
    'delay', 'jitter', 'interArrival', 'seqGap', 'pktSize', 'pktRate', 'pdr', 'fwdRatio',
    'numNeighbors', 'neighborDensity', 'cbr', 'rxMsgRate', 'sameLocCount', 'distEst',
    'posRadioMismatch', 'impliedSpeed', 'speedDelta', 'idAge', 'txPower'
]

POS_COLS = ['posX', 'posY', 'rxPosX', 'rxPosY']
WITHOUT_POS_FEATURES = [c for c in RAW_36_FEATURES if c not in POS_COLS]

CLASS_NAMES = ['Benign', 'Blackhole', 'DDoS', 'Sybil']
CLASS_MAP = {'Benign': 0, 'Blackhole': 1, 'DDoS': 2, 'Sybil': 3}

TRAIN_FILES = ['train_1.csv', 'train_2.csv', 'train_3.csv', 'train_4.csv']
VAL_FILES = ['train_5.csv', 'train_6.csv']
TEST_FILES = {
    'val_7 (LLM)': 'val_7.csv',
    'test_8': 'test_8.csv',
    'test_9': 'test_9.csv',
    'test_hard_10': 'test_hard_10.csv'
}

def load_data(files, features):
    print(f"Loading {files} with {len(features)} features...", flush=True)
    dfs = []
    cols_to_load = features + ['label', 'attackType']
    for f in files:
        df = pd.read_csv(f, usecols=cols_to_load)
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    X = full_df[features].values.astype(np.float32)
    y_bin = full_df['label'].values.astype(np.int64)
    y_multi = full_df['attackType'].map(CLASS_MAP).values.astype(np.int64)
    return X, y_bin, y_multi

def evaluate_model(clf, X, y_multi, y_bin, dataset_name):
    preds_proba = clf.predict_proba(X)
    preds = np.argmax(preds_proba, axis=1)
    
    # Per-class metrics
    p, r, f1, sup = precision_recall_fscore_support(y_multi, preds, labels=[0, 1, 2, 3], zero_division=0)
    macro_f1 = f1_score(y_multi, preds, average='macro', zero_division=0)
    
    # Binary predictions (class != 0 is malicious)
    bin_preds = (preds > 0).astype(int)
    bin_f1 = f1_score(y_bin, bin_preds, zero_division=0)
    bin_probs = 1.0 - preds_proba[:, 0]
    try:
        pr_auc = average_precision_score(y_bin, bin_probs)
    except:
        pr_auc = np.nan
        
    res = {
        'macro_f1': macro_f1,
        'bin_f1': bin_f1,
        'pr_auc': pr_auc,
        'per_class_f1': {CLASS_NAMES[i]: f1[i] for i in range(4)},
        'per_class_rec': {CLASS_NAMES[i]: r[i] for i in range(4)},
        'per_class_prec': {CLASS_NAMES[i]: p[i] for i in range(4)},
        'support': {CLASS_NAMES[i]: sup[i] for i in range(4)}
    }
    return res

def run_experiment(features, name="WITH_POSITIONS"):
    print("\n" + "=" * 80)
    print(f"LIGHTGBM BASELINE - EXPERIMENT: {name} ({len(features)} features)")
    print("=" * 80)
    
    # 1. Load train and val
    X_train, y_train_bin, y_train_multi = load_data(TRAIN_FILES, features)
    X_val, y_val_bin, y_val_multi = load_data(VAL_FILES, features)
    
    print(f"Train rows: {len(X_train):,}, Val rows: {len(X_val):,}")
    
    # Compute class weights for 4-class multi-class
    counts = np.bincount(y_train_multi, minlength=4)
    total = len(y_train_multi)
    # Balanced class weights
    class_weights = total / (4.0 * counts)
    print(f"Class distribution in train: {dict(zip(CLASS_NAMES, counts))}")
    print(f"Computed balanced class weights: {dict(zip(CLASS_NAMES, np.round(class_weights, 3)))}")
    
    sample_weights = class_weights[y_train_multi]
    
    # LightGBM Classifier
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
        random_state=42,
        importance_type='gain'
    )
    
    print("Training LightGBM...", flush=True)
    clf.fit(
        X_train, y_train_multi,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val_multi)],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False), lgb.log_evaluation(period=50)]
    )
    
    # Top-15 Feature Importances
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\n--- TOP 15 FEATURE IMPORTANCES (GAIN) [{name}] ---")
    for rank, idx in enumerate(sorted_idx[:15], 1):
        print(f"  {rank:2d}. {features[idx]:<22}: {importances[idx]:14,.2f}")
        
    # Evaluate on Validation
    val_res = evaluate_model(clf, X_val, y_val_multi, y_val_bin, "Validation (train_5 + train_6)")
    print(f"\n--- VALIDATION RESULTS [{name}] ---")
    print(f"Macro-F1: {val_res['macro_f1']:.4f} | Binary F1: {val_res['bin_f1']:.4f} | PR-AUC: {val_res['pr_auc']:.4f}")
    for c in CLASS_NAMES:
        print(f"  {c:<12}: P={val_res['per_class_prec'][c]:.4f}, R={val_res['per_class_rec'][c]:.4f}, F1={val_res['per_class_f1'][c]:.4f} (Support={val_res['support'][c]:,})")
        
    # Check for near-perfect leakage
    if val_res['macro_f1'] > 0.99:
        print("\n" + "!" * 80)
        print("WARNING: Macro-F1 > 0.99 detected! Possible feature leakage!")
        print("!" * 80)
        
    # Evaluate on Eval/Test files
    test_results = {}
    for dname, fname in TEST_FILES.items():
        X_test, y_test_bin, y_test_multi = load_data([fname], features)
        res = evaluate_model(clf, X_test, y_test_multi, y_test_bin, dname)
        test_results[dname] = res
        print(f"\n--- {dname} RESULTS [{name}] ---")
        print(f"Macro-F1: {res['macro_f1']:.4f} | Binary F1: {res['bin_f1']:.4f} | PR-AUC: {res['pr_auc']:.4f}")
        for c in CLASS_NAMES:
            print(f"  {c:<12}: P={res['per_class_prec'][c]:.4f}, R={res['per_class_rec'][c]:.4f}, F1={res['per_class_f1'][c]:.4f} (Support={res['support'][c]:,})")
            
    return val_res, test_results, clf

if __name__ == '__main__':
    # Experiment 1: With all 36 features (including absolute positions)
    val_with_pos, test_with_pos, clf_with = run_experiment(RAW_36_FEATURES, "WITH_POSITIONS")
    
    # Experiment 2: Without absolute positions (32 features)
    val_no_pos, test_no_pos, clf_no = run_experiment(WITHOUT_POS_FEATURES, "WITHOUT_POSITIONS")
    
    print("\n" + "=" * 80)
    print("ABLATION SUMMARY: WITH POSITIONS vs WITHOUT POSITIONS")
    print("=" * 80)
    print(f"{'Dataset':<20} | {'Macro-F1 (With Pos)':<20} | {'Macro-F1 (No Pos)':<20} | {'Blackhole F1 (With)':<20} | {'Blackhole F1 (No)':<20}")
    print("-" * 105)
    print(f"{'Val (train_5+6)':<20} | {val_with_pos['macro_f1']:<20.4f} | {val_no_pos['macro_f1']:<20.4f} | {val_with_pos['per_class_f1']['Blackhole']:<20.4f} | {val_no_pos['per_class_f1']['Blackhole']:<20.4f}")
    for dname in TEST_FILES:
        print(f"{dname:<20} | {test_with_pos[dname]['macro_f1']:<20.4f} | {test_no_pos[dname]['macro_f1']:<20.4f} | {test_with_pos[dname]['per_class_f1']['Blackhole']:<20.4f} | {test_no_pos[dname]['per_class_f1']['Blackhole']:<20.4f}")
    print("=" * 80)
