# VANET Intrusion Detection Benchmark (V2X Beacons)

This repository contains a complete, robust Intrusion Detection System (IDS) and benchmark pipeline for Vehicle-to-Everything (V2X) beacon logs. The dataset consists of 10 large CSV files (~10M rows total) capturing vehicle kinematics, radio measurements, packet statistics, and plausible security anomalies.

---

## 1. System Architecture & Methodology

### A. Label Structure & Class Mapping
Audit of all 10 CSV files verified exactly 4 classes in the dataset:
- **0 - Benign** (83.18%)
- **1 - Blackhole** (1.30%) — Minority attack, dropped packets
- **2 - DDoS** (5.68%) — High-rate burst message flooding
- **3 - Sybil** (9.84%) — Impersonation / dense phantom vehicle generation

### B. Preprocessing & Leakage Prevention
- **Strict File-Level Splits**:
  - **Train**: `train_1.csv` ... `train_4.csv` (~1.6M rows)
  - **Validation**: `train_5.csv` + `train_6.csv` (~800k rows)
  - **Evaluation (Separate, Never Pooled)**:
    - `val_7.csv` (LLM-generated / adversarial synthetic test set)
    - `test_8.csv` (Unseen test scenario 1)
    - `test_9.csv` (Unseen test scenario 2)
    - `test_hard_10.csv` (High-congestion stress test)
- **Feature Hygiene**:
  - Dropped identifiers and labels: `label`, `attackType`, `attackerNode`, `runId`, `receiverId`, `senderId`, `simTime`, `msgSeq`.
  - Dropped constant columns across all files: `rsrq` (= -3.0).
  - Dropped absolute position coordinates: `posX`, `posY`, `rxPosX`, `rxPosY` as the default setting (proven by spatial overlap audits to overfit and hurt OOD generalization; ablation verifies this).
  - Heavy-tail features (`pktRate`, `cbr`, `delay`, `pathLoss`, etc.) transformed via $\log(1 + x)$.
  - Scaled using `RobustScaler` (median & IQR) fitted strictly on training data.
- **Sequential Feature Engineering**:
  - Consecutive beacon kinematics per `(runId, receiverId, senderId)`: $\Delta t$, `posJump`, `posJumpVsImplied`, `speedDelta_seq`, `headingDelta_seq`.

### C. Models Evaluated
1. **LightGBM Row-Level**: Gradient boosted trees on individual beacon records (32 non-position features).
2. **LightGBM Windowed**: Gradient boosted trees on 185 rolling window statistics (mean, std, min, max, last value) computed over a 20-beacon window.
3. **LSTM Models ($W=10, 20, 40$)**: 2-layer Bi/Unidirectional LSTM with 128 hidden units, dropout (0.3), dual-head prediction (Binary intrusion + 4-class multi-head).
4. **Spatial Ablation ($W=20$ With Positions)**: Evaluates impact of keeping raw coordinate features.
5. **Context Ablation ($W=1$)**: 1-step window (no temporal context).

---

## 2. Environment & Requirements

- **OS**: Windows 11
- **Python**: Python 3.12.7 (via Anaconda)
- **PyTorch**: 2.9.1+cpu (Multi-threading enabled, 16 threads)
- **Dependencies**: `lightgbm`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `pyyaml`

---

## 3. Reproduction Commands

All commands should be executed from the project root using Windows PowerShell:

```powershell
# 1. Run Dataset Audit
& "C:\Users\harsh\anaconda3\python.exe" work\run_audit.py

# 2. Analyze Attack Episode Durations
& "C:\Users\harsh\anaconda3\python.exe" work\attack_episodes.py

# 3. Investigate Trajectory Overlap & Leakage
& "C:\Users\harsh\anaconda3\python.exe" work\investigate_runs_trajectories.py

# 4. Run Full Training & Benchmark Pipeline
# Trains LightGBM (Row-level & Windowed) and LSTM (W=10, 20, 40, W=20 with pos, W=1)
& "C:\Users\harsh\anaconda3\python.exe" work\train.py
```

---

## 4. Directory Structure

```text
VANET_Intrusion_Detection/
├── work/
│   ├── run_audit.py                 # Dataset inspection and column verification
│   ├── attack_episodes.py           # Consecutive beacon episode duration statistics
│   ├── investigate_runs_trajectories.py # Trajectory overlap and mid-run attack analysis
│   ├── preprocess.py                # Robust scaler and sliding window pipeline
│   ├── models.py                    # MultiHeadLSTM PyTorch model definition
│   ├── evaluate.py                  # Evaluation metrics & confusion matrix generation
│   ├── train.py                     # Main multi-model training and evaluation script
│   ├── config.yaml                  # Experiment configuration
│   ├── audit_report.txt             # Saved dataset audit summary
│   ├── checkpoints/                 # Best model weights and epoch checkpoints
│   ├── cache/                       # Cached .npz sliding window arrays
│   └── results/
│       ├── metrics.json             # Exhaustive metrics across all models and splits
│       ├── comparison.md            # Benchmark comparison table
│       └── confusion_*.png          # Normalized confusion matrices per model & dataset
└── README.md
```
