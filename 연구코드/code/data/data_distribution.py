import pandas as pd
import numpy as np

FILE_PATH = r"C:\Users\jhjun\OneDrive\Desktop\2026-1_Study\연구코드\data\강남\aec\강남_z_bounds.xlsx"
COLS = ["slices", "z_range"]

df = pd.read_excel(FILE_PATH, usecols=COLS)

# ── 기본 결측 확인 ─────────────────────────────────────────────
print("=" * 55)
print("  Missing Values")
print("=" * 55)
print(df[COLS].isnull().sum().rename("n_missing").to_frame())

# ── 기술통계 ──────────────────────────────────────────────────
stats_funcs = {
    "N":        lambda s: s.count(),
    "Mean":     lambda s: s.mean(),
    "SD":       lambda s: s.std(ddof=1),
    "Min":      lambda s: s.min(),
    "P25":      lambda s: s.quantile(0.25),
    "Median":   lambda s: s.median(),
    "P75":      lambda s: s.quantile(0.75),
    "Max":      lambda s: s.max(),
    "IQR":      lambda s: s.quantile(0.75) - s.quantile(0.25),
    "Skewness": lambda s: s.skew(),
    "Kurtosis": lambda s: s.kurt(),
}

results = {}
for col in COLS:
    s = df[col].dropna()
    results[col] = {k: fn(s) for k, fn in stats_funcs.items()}

stat_df = pd.DataFrame(results).T

print("\n" + "=" * 55)
print("  Descriptive Statistics")
print("=" * 55)
fmt = {
    "N":        "{:.0f}",
    "Mean":     "{:.2f}",
    "SD":       "{:.2f}",
    "Min":      "{:.1f}",
    "P25":      "{:.1f}",
    "Median":   "{:.1f}",
    "P75":      "{:.1f}",
    "Max":      "{:.1f}",
    "IQR":      "{:.1f}",
    "Skewness": "{:.3f}",
    "Kurtosis": "{:.3f}",
}
for col_name, f in fmt.items():
    stat_df[col_name] = stat_df[col_name].map(lambda x, f=f: f.format(x))

print(stat_df.to_string())

# ── Outlier 탐지 (IQR fence) ──────────────────────────────────
print("\n" + "=" * 55)
print("  Outlier Count  (IQR × 1.5 rule)")
print("=" * 55)
for col in COLS:
    s = df[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((s < lo) | (s > hi)).sum()
    pct = n_out / len(s) * 100
    print(f"  {col:10s}: {n_out:4d} outliers ({pct:.1f}%)  "
          f"[fence: {lo:.1f} – {hi:.1f}]")

# ── slices vs z_range 상관 ────────────────────────────────────
valid = df[COLS].dropna()
r = valid["slices"].corr(valid["z_range"])
print(f"\n  Pearson r (slices vs z_range): {r:.4f}")
print("=" * 55)