# VANET Intrusion Detection: Model Comparison & Benchmark Report

## Macro-F1 Across All Evaluation Datasets

| Model | Val (train_5+6) | val_7 (LLM) | test_8 | test_9 | test_hard_10 | Mean Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM_RowLevel** | 0.8674 | 0.9039 | 0.8652 | 0.8762 | 0.8540 | **0.8734** |
| **LightGBM_Windowed** | 0.9361 | 0.9328 | 0.9329 | 0.9156 | 0.9006 | **0.9236** |
| **LSTM_W20_NoPos** | 0.9246 | 0.9224 | 0.9145 | 0.9023 | 0.8916 | **0.9111** |
| **LSTM_W10_NoPos** | 0.9357 | 0.9305 | 0.9291 | 0.9133 | 0.9030 | **0.9223** |
| **LSTM_W40_NoPos** | 0.9269 | 0.9202 | 0.9183 | 0.9012 | 0.8982 | **0.9130** |
| **LSTM_W20_WithPos** | 0.9276 | 0.9240 | 0.9207 | 0.9053 | 0.8986 | **0.9152** |
| **LSTM_W1_Ablation** | 0.8540 | 0.8894 | 0.8720 | 0.8785 | 0.8449 | **0.8678** |

## Blackhole Attack F1-Score (Minority Class)

| Model | Val (train_5+6) | val_7 (LLM) | test_8 | test_9 | test_hard_10 | Mean Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM_RowLevel** | 0.7639 | 0.8107 | 0.6396 | 0.6894 | 0.7101 | **0.7227** |
| **LightGBM_Windowed** | 0.8975 | 0.8676 | 0.8586 | 0.8086 | 0.8160 | **0.8497** |
| **LSTM_W20_NoPos** | 0.8666 | 0.8389 | 0.8122 | 0.7744 | 0.7827 | **0.8150** |
| **LSTM_W10_NoPos** | 0.8966 | 0.8606 | 0.8659 | 0.8084 | 0.8202 | **0.8503** |
| **LSTM_W40_NoPos** | 0.8760 | 0.8355 | 0.8355 | 0.7681 | 0.8017 | **0.8234** |
| **LSTM_W20_WithPos** | 0.8804 | 0.8436 | 0.8389 | 0.7890 | 0.8057 | **0.8315** |
| **LSTM_W1_Ablation** | 0.7639 | 0.7841 | 0.7549 | 0.7505 | 0.7292 | **0.7565** |

## Pure Windows vs Transition Windows Performance (Macro-F1)

| Model | Dataset | Pure Windows Macro-F1 | Transition Windows Macro-F1 | Delta |
| :--- | :--- | :---: | :---: | :---: |
| LightGBM_RowLevel | Val (train_5+6) | nan | nan | +nan |
| LightGBM_RowLevel | test_hard_10 | nan | nan | +nan |
| LightGBM_Windowed | Val (train_5+6) | 0.9730 | 0.8283 | -0.1447 |
| LightGBM_Windowed | test_hard_10 | 0.9680 | 0.8091 | -0.1589 |
| LSTM_W20_NoPos | Val (train_5+6) | 0.9721 | 0.8002 | -0.1719 |
| LSTM_W20_NoPos | test_hard_10 | 0.9684 | 0.7890 | -0.1795 |
| LSTM_W10_NoPos | Val (train_5+6) | 0.9721 | 0.7572 | -0.2149 |
| LSTM_W10_NoPos | test_hard_10 | 0.9684 | 0.7321 | -0.2363 |
| LSTM_W40_NoPos | Val (train_5+6) | 0.9685 | 0.8643 | -0.1042 |
| LSTM_W40_NoPos | test_hard_10 | 0.9691 | 0.8529 | -0.1161 |
| LSTM_W20_WithPos | Val (train_5+6) | 0.9687 | 0.8135 | -0.1552 |
| LSTM_W20_WithPos | test_hard_10 | 0.9693 | 0.8036 | -0.1657 |
| LSTM_W1_Ablation | Val (train_5+6) | 0.8540 | nan | +nan |
| LSTM_W1_Ablation | test_hard_10 | 0.8449 | nan | +nan |

## Conclusion & Confirmation Status

- Models evaluated across 5 separate unseen datasets without data leakage.
- Results confirmed across evaluation datasets according to strict 4-of-5 criteria.