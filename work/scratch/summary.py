import json
import numpy as np

with open('work/results/metrics.json') as f:
    d = json.load(f)

models = ['LightGBM_RowLevel', 'LightGBM_Windowed', 'LSTM_W10_NoPos', 'LSTM_W20_NoPos', 'LSTM_W40_NoPos', 'LSTM_W20_WithPos', 'LSTM_W1_Ablation']
dsets = ['val', 'val_7_llm', 'test_8', 'test_9', 'test_hard_10']

print("=== MACRO-F1 ===")
for m in models:
    scores = [d[m][ds]['macro_f1'] for ds in dsets]
    print(f"{m:<20}: {' '.join([f'{s:.4f}' for s in scores])} | Mean: {np.mean(scores):.4f}")

print("\n=== BLACKHOLE PRECISION & RECALL ===")
for m in models:
    print(f"\n{m}:")
    for ds in dsets:
        bh = d[m][ds]['blackhole']
        print(f"  {ds:<12}: P={bh['precision']:.4f}, R={bh['recall']:.4f}, F1={bh['f1']:.4f}, PR-AUC={bh['pr_auc']:.4f}")

print("\n=== PER-CLASS PR-AUC ON TEST_HARD_10 ===")
for m in models:
    pc = d[m]['test_hard_10']['per_class']
    pr_str = ' | '.join([f"{c}: {pc[c]['pr_auc']:.4f}" for c in pc])
    print(f"{m:<20} | {pr_str}")

print("\n=== PURE VS TRANSITION (TEST_HARD_10) ===")
for m in models:
    th = d[m]['test_hard_10']
    pure = th.get('pure_windows', {}).get('macro_f1', np.nan)
    trans = th.get('transition_windows', {}).get('macro_f1', np.nan)
    print(f"{m:<20} | Pure: {pure:.4f} | Transition: {trans:.4f} | Delta: {trans-pure:+.4f}")
