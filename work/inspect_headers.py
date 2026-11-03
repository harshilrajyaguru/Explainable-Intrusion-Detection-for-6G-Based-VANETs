import os
import glob
import pandas as pd

files = [
    'train_1.csv', 'train_2.csv', 'train_3.csv', 'train_4.csv', 'train_5.csv', 'train_6.csv',
    'val_7.csv', 'test_8.csv', 'test_9.csv', 'test_hard_10.csv'
]

print("Checking files...")
all_cols = []
for f in files:
    if not os.path.exists(f):
        print(f"Missing file: {f}")
        continue
    df = pd.read_csv(f, nrows=5)
    cols = df.columns.tolist()
    all_cols.append((f, cols))
    print(f"{f}: {len(cols)} columns")

base_cols = all_cols[0][1]
for f, cols in all_cols[1:]:
    if cols != base_cols:
        print(f"Column mismatch in {f}!")
        diff = set(cols).symmetric_difference(set(base_cols))
        print("Diff:", diff)
    else:
        print(f"{f}: matches train_1.csv columns exactly.")
