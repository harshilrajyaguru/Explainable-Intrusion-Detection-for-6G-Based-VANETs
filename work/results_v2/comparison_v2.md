# VANET Intrusion Detection Benchmark (V2): Multi-Seed, Bootstrap CIs & Latency Analysis

## 1. Macro-F1 Performance (Mean ± Std over 5 Seeds & Bootstrap 95% CI)

| Model | Val (train_5+6) | val_7 (LLM) | test_8 | test_9 | test_hard_10 | Mean Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM_Windowed** | 0.9358±0.0002<br>[0.9252, 0.9444] | 0.9318±0.0011<br>[0.9172, 0.9455] | 0.9324±0.0009<br>[0.9142, 0.9462] | 0.9163±0.0004<br>[0.8990, 0.9304] | 0.9008±0.0002<br>[0.8910, 0.9096] | **0.9234** |
| **LSTM_ManyToMany** | 0.9107±0.0041<br>[0.8927, 0.9243] | 0.9311±0.0014<br>[0.9168, 0.9454] | 0.9233±0.0078<br>[0.9057, 0.9402] | 0.9126±0.0024<br>[0.8953, 0.9280] | 0.8920±0.0017<br>[0.8805, 0.9029] | **0.9140** |
| **LSTM_W10_NoPos** | 0.9267±0.0040<br>[0.9221, 0.9415] | 0.9297±0.0021<br>[0.9194, 0.9454] | 0.9209±0.0044<br>[0.9038, 0.9381] | 0.9141±0.0025<br>[0.9015, 0.9314] | 0.8988±0.0030<br>[0.8935, 0.9094] | **0.9180** |
| **LSTM_W40_NoPos** | 0.9286±0.0034<br>[0.9199, 0.9429] | 0.9225±0.0023<br>[0.9038, 0.9384] | 0.9156±0.0053<br>[0.8845, 0.9302] | 0.9049±0.0014<br>[0.8837, 0.9226] | 0.8997±0.0023<br>[0.8912, 0.9115] | **0.9142** |

## 2. Blackhole Attack F1-Score (Mean ± Std & 95% CI)

| Model | Val (train_5+6) | val_7 (LLM) | test_8 | test_9 | test_hard_10 | Mean Blackhole F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM_Windowed** | 0.8975±0.0000<br>[0.8606, 0.9254] | 0.8671±0.0010<br>[0.8101, 0.9121] | 0.8577±0.0011<br>[0.7837, 0.9062] | 0.8086±0.0000<br>[0.7417, 0.8630] | 0.8158±0.0005<br>[0.7825, 0.8455] | **0.8494** |
| **LSTM_ManyToMany** | 0.8905±0.0035<br>[0.8575, 0.9213] | 0.8605±0.0034<br>[0.8026, 0.9056] | 0.8372±0.0297<br>[0.7651, 0.8935] | 0.7984±0.0096<br>[0.7321, 0.8528] | 0.8129±0.0015<br>[0.7812, 0.8443] | **0.8399** |
| **LSTM_W10_NoPos** | 0.8846±0.0079<br>[0.8549, 0.9192] | 0.8576±0.0068<br>[0.8137, 0.9085] | 0.8406±0.0110<br>[0.7649, 0.8935] | 0.8089±0.0070<br>[0.7586, 0.8647] | 0.8154±0.0065<br>[0.7874, 0.8428] | **0.8414** |
| **LSTM_W40_NoPos** | 0.8794±0.0013<br>[0.8338, 0.9126] | 0.8411±0.0068<br>[0.7639, 0.8851] | 0.8251±0.0203<br>[0.6939, 0.8724] | 0.7755±0.0063<br>[0.6786, 0.8310] | 0.8056±0.0016<br>[0.7681, 0.8388] | **0.8253** |

## 3. Statistical Significance Analysis (Bootstrap CI Separation)

Rule: Differences are designated **Statistically Significant** if and only if their 95% Confidence Intervals are disjoint (separate).

| Comparison | Dataset | Model A (95% CI) | Model B (95% CI) | CIs Separate? | Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: |
| LightGBM_Windowed vs LSTM_W10_NoPos | Val (train_5+6) | [0.9252, 0.9444] | [0.9221, 0.9415] | NO | NO (Overlapping) |
| LightGBM_Windowed vs LSTM_W10_NoPos | test_hard_10 | [0.8910, 0.9096] | [0.8935, 0.9094] | NO | NO (Overlapping) |
| LightGBM_Windowed vs LSTM_W40_NoPos | Val (train_5+6) | [0.9252, 0.9444] | [0.9199, 0.9429] | NO | NO (Overlapping) |
| LightGBM_Windowed vs LSTM_W40_NoPos | test_hard_10 | [0.8910, 0.9096] | [0.8912, 0.9115] | NO | NO (Overlapping) |
| LSTM_W10_NoPos vs LSTM_ManyToMany | Val (train_5+6) | [0.9221, 0.9415] | [0.8927, 0.9243] | NO | NO (Overlapping) |
| LSTM_W10_NoPos vs LSTM_ManyToMany | test_hard_10 | [0.8935, 0.9094] | [0.8805, 0.9029] | NO | NO (Overlapping) |
| LightGBM_Windowed vs LSTM_ManyToMany | Val (train_5+6) | [0.9252, 0.9444] | [0.8927, 0.9243] | YES | **YES** (Significant) |
| LightGBM_Windowed vs LSTM_ManyToMany | test_hard_10 | [0.8910, 0.9096] | [0.8805, 0.9029] | NO | NO (Overlapping) |

## 4. Time-to-Detect (TTD) After Attack Onset

Evaluated on unseen test scenarios (`test_8` and `test_hard_10`). Reported as Median / 90th percentile (seconds & beacons).

| Model | Attack Type | Detection Rate | TTD Median (s) | TTD p90 (s) | TTD Median (beacons) | TTD p90 (beacons) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| LightGBM_Windowed | **Blackhole** | 7.5% | 2.40s | 12.60s | 7 | 20 |
| LightGBM_Windowed | **DDoS** | 26.8% | 0.20s | 1.11s | 6 | 10 |
| LightGBM_Windowed | **Sybil** | 29.0% | 1.90s | 10.20s | 9 | 20 |
| LSTM_ManyToMany | **Blackhole** | 8.1% | 1.95s | 12.15s | 7 | 20 |
| LSTM_ManyToMany | **DDoS** | 27.3% | 0.20s | 1.10s | 6 | 10 |
| LSTM_ManyToMany | **Sybil** | 30.7% | 1.80s | 10.10s | 9 | 20 |
| LSTM_W10_NoPos | **Blackhole** | 8.3% | 2.40s | 9.80s | 7 | 10 |
| LSTM_W10_NoPos | **DDoS** | 27.9% | 0.20s | 1.10s | 6 | 10 |
| LSTM_W10_NoPos | **Sybil** | 31.7% | 1.20s | 7.11s | 9 | 10 |
| LSTM_W40_NoPos | **Blackhole** | 7.0% | 2.80s | 14.20s | 8 | 40 |
| LSTM_W40_NoPos | **DDoS** | 24.9% | 0.20s | 1.16s | 6 | 13 |
| LSTM_W40_NoPos | **Sybil** | 27.6% | 2.35s | 13.40s | 10 | 40 |

## 5. False-Alarm Rate (FAR) on Benign Traffic

| Model | Val (alarms/min) | val_7 (alarms/min) | test_8 (alarms/min) | test_9 (alarms/min) | test_hard_10 (alarms/min) | Mean FAR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM_Windowed** | 0.08 | 0.21 | 0.09 | 0.15 | 0.15 | **0.14** |
| **LSTM_ManyToMany** | 0.38 | 0.28 | 0.16 | 0.25 | 0.40 | **0.29** |
| **LSTM_W10_NoPos** | 0.14 | 0.26 | 0.18 | 0.24 | 0.23 | **0.21** |
| **LSTM_W40_NoPos** | 0.18 | 0.30 | 0.20 | 0.28 | 0.24 | **0.24** |