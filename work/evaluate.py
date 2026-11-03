import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support, f1_score, accuracy_score,
    precision_score, recall_score, roc_auc_score, average_precision_score,
    confusion_matrix
)

sys.stdout.reconfigure(line_buffering=True)

CLASS_NAMES = ['Benign', 'Blackhole', 'DDoS', 'Sybil']

def tune_multiclass_thresholds(y_true, probas):
    """
    Tunes per-class threshold multipliers on validation set to maximize Macro-F1.
    Default prediction is argmax(probas).
    Adjusted prediction is argmax(probas / thresholds).
    """
    best_macro_f1 = f1_score(y_true, np.argmax(probas, axis=1), average='macro', zero_division=0)
    best_multipliers = np.ones(4, dtype=np.float32)
    
    # Simple coordinate search around 1.0 for each minority class
    search_vals = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    for c in [1, 2, 3]: # Blackhole, DDoS, Sybil
        best_val = best_multipliers[c]
        for val in search_vals:
            current_mults = best_multipliers.copy()
            current_mults[c] = val
            adj_probs = probas * current_mults
            preds = np.argmax(adj_probs, axis=1)
            score = f1_score(y_true, preds, average='macro', zero_division=0)
            if score > best_macro_f1:
                best_macro_f1 = score
                best_val = val
        best_multipliers[c] = best_val
        
    print(f"Tuned probability multipliers (relative to 1.0): {dict(zip(CLASS_NAMES, np.round(best_multipliers, 2)))}")
    print(f"Validation Macro-F1 with tuned thresholds: {best_macro_f1:.4f}")
    return best_multipliers

def compute_all_metrics(y_true_m, y_true_b, probas_m, probas_b, is_pure=None, multipliers=None):
    if multipliers is not None:
        adj_probas = probas_m * multipliers
        preds_m = np.argmax(adj_probas, axis=1)
    else:
        preds_m = np.argmax(probas_m, axis=1)
        
    preds_b = np.argmax(probas_b, axis=1) if probas_b.ndim == 2 else (probas_b >= 0.5).astype(int)
    bin_score = probas_b[:, 1] if probas_b.ndim == 2 else probas_b
    
    acc = accuracy_score(y_true_m, preds_m)
    macro_f1 = f1_score(y_true_m, preds_m, average='macro', zero_division=0)
    bin_f1 = f1_score(y_true_b, preds_b, zero_division=0)
    
    try:
        pr_auc_bin = average_precision_score(y_true_b, bin_score)
    except:
        pr_auc_bin = np.nan
        
    # Per-class metrics
    p, r, f1, sup = precision_recall_fscore_support(y_true_m, preds_m, labels=[0, 1, 2, 3], zero_division=0)
    
    # Per-class PR-AUC
    pr_auc_per_class = {}
    for c in range(4):
        y_c = (y_true_m == c).astype(int)
        score_c = probas_m[:, c]
        try:
            pr_auc_per_class[CLASS_NAMES[c]] = float(average_precision_score(y_c, score_c))
        except:
            pr_auc_per_class[CLASS_NAMES[c]] = np.nan
            
    res = {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'binary_f1': float(bin_f1),
        'binary_pr_auc': float(pr_auc_bin),
        'per_class': {
            CLASS_NAMES[i]: {
                'precision': float(p[i]),
                'recall': float(r[i]),
                'f1': float(f1[i]),
                'support': int(sup[i]),
                'pr_auc': pr_auc_per_class[CLASS_NAMES[i]]
            } for i in range(4)
        },
        'blackhole': {
            'precision': float(p[1]),
            'recall': float(r[1]),
            'f1': float(f1[1]),
            'support': int(sup[1]),
            'pr_auc': pr_auc_per_class['Blackhole']
        }
    }
    
    # Pure vs Transition metrics
    if is_pure is not None:
        pure_mask = (is_pure == True)
        trans_mask = (is_pure == False)
        
        if np.sum(pure_mask) > 0:
            pure_acc = accuracy_score(y_true_m[pure_mask], preds_m[pure_mask])
            pure_macro = f1_score(y_true_m[pure_mask], preds_m[pure_mask], average='macro', zero_division=0)
            res['pure_windows'] = {
                'count': int(np.sum(pure_mask)),
                'accuracy': float(pure_acc),
                'macro_f1': float(pure_macro)
            }
        if np.sum(trans_mask) > 0:
            trans_acc = accuracy_score(y_true_m[trans_mask], preds_m[trans_mask])
            trans_macro = f1_score(y_true_m[trans_mask], preds_m[trans_mask], average='macro', zero_division=0)
            res['transition_windows'] = {
                'count': int(np.sum(trans_mask)),
                'accuracy': float(trans_acc),
                'macro_f1': float(trans_macro)
            }
            
    return res, preds_m

def plot_and_save_confusion_matrix(y_true, y_pred, model_name, dataset_name, out_dir="work/results"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    
    # Save CSV
    safe_mname = model_name.replace(" ", "_").replace("=", "").replace("/", "_")
    safe_dname = dataset_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")
    csv_file = os.path.join(out_dir, f"confusion_{safe_mname}_{safe_dname}.csv")
    df_cm = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    df_cm.to_csv(csv_file)
    
    # Save PNG
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix: {model_name} on {dataset_name}")
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.tight_layout()
    png_file = os.path.join(out_dir, f"confusion_{safe_mname}_{safe_dname}.png")
    plt.savefig(png_file, dpi=150)
    plt.close()
    return csv_file, png_file
