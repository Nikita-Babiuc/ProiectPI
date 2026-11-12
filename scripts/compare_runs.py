import sys

import pandas as pd

# Shows how to use the script when no file is given.
if len(sys.argv) < 2:
    sys.exit("usage: python scripts/compare_runs.py run1.csv [run2.csv ...]")

for path in sys.argv[1:]:
    # Skips a missing file instead of stopping with a traceback.
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"{path}: file not found")
        continue
    df = df[df["time_s"] > 15]  # skips the warm-up
    if df.empty:
        print(f"{path}: no data after the 15 s warm-up")
        continue
    # The SNR median only counts rows that have an estimate, so read it together with valid.
    print(f"{path}: median SNR {df['snr_db'].median():.1f} dB, "
          f"valid {df['valid'].mean():.0%}, bpm mean {df['bpm'].mean():.1f}, bpm std {df['bpm'].std():.1f}")