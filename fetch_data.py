import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta

# ── period from env (set by workflow_dispatch input) ────────────────────────
_period_env = os.environ.get("TRADING_PERIOD", "252").strip()
try:
    TARGET_DAYS = int(_period_env)
except ValueError:
    TARGET_DAYS = 252

ALLOWED_PERIODS = [30, 65, 90, 128, 252]
if TARGET_DAYS not in ALLOWED_PERIODS:
    print(f"WARNING: {TARGET_DAYS} not in allowed periods {ALLOWED_PERIODS}, defaulting to 252")
    TARGET_DAYS = 252

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "stocks.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "corr_data.json")

MIN_DAYS   = max(20, int(TARGET_DAYS * 0.80))
FETCH_DAYS = max(540, TARGET_DAYS * 3)

print(f"Target period : {TARGET_DAYS} trading days")
print(f"Min days      : {MIN_DAYS}")
print(f"Fetch span    : {FETCH_DAYS} calendar days")

# ── load stock list ──────────────────────────────────────────────────────────
if not os.path.exists(CSV_PATH):
    print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# Support both column naming conventions
if "Symbol" in df.columns and "Industry" in df.columns:
    df["Symbol"]       = df["Symbol"].str.strip()
    df["Industry"]     = df["Industry"].str.strip()
    if "Company Name" in df.columns:
        df["Company Name"] = df["Company Name"].str.strip()
    else:
        df["Company Name"] = df["Symbol"]
elif "stock" in df.columns and "sector" in df.columns:
    df = df.rename(columns={"stock": "Symbol", "sector": "Industry"})
    df["Symbol"]       = df["Symbol"].str.strip()
    df["Industry"]     = df["Industry"].str.strip()
    df["Company Name"] = df["Symbol"]
else:
    print(f"ERROR: CSV must have columns 'Symbol'+'Industry' or 'stock'+'sector'. Found: {list(df.columns)}", file=sys.stderr)
    sys.exit(1)

# ── IMPORTANT: keep all (Symbol, Industry) pairs — a stock can belong to
#    multiple indices. Deduplicate only exact duplicate rows.
df = df[["Company Name","Industry","Symbol"]].dropna()
df = df.drop_duplicates(subset=["Symbol","Industry"])   # remove exact dupes only

# Build unique ticker list for price download (each ticker downloaded once)
unique_symbols = df["Symbol"].unique().tolist()
tickers        = [s + ".NS" for s in unique_symbols]
print(f"Loaded {len(df)} stock-industry rows, {len(unique_symbols)} unique symbols from CSV")

# ── fetch prices ─────────────────────────────────────────────────────────────
end   = datetime.today() + timedelta(days=1)  # +1 so yfinance includes today (end is exclusive)
start = end - timedelta(days=FETCH_DAYS)
print(f"Downloading {len(tickers)} tickers ({start.date()} to {end.date()}) ...")

raw = yf.download(
    tickers,
    start=start.strftime("%Y-%m-%d"),
    end=end.strftime("%Y-%m-%d"),
    interval="1d",
    auto_adjust=True,
    progress=False,
    threads=True,
)

# ── extract Close panel ───────────────────────────────────────────────────────
if isinstance(raw.columns, pd.MultiIndex):
    close = raw["Close"]
else:
    close = raw[["Close"]]

print(f"Raw price shape: {close.shape}")

# ── identify TARGET_DAYS most-recent trading dates ───────────────────────────
trading_dates = close.index[close.notna().sum(axis=1) >= 10]
if len(trading_dates) < TARGET_DAYS:
    raise ValueError(
        f"Only {len(trading_dates)} common trading dates found, need {TARGET_DAYS}."
    )

last_n_dates = trading_dates[-TARGET_DAYS:]
close_n      = close.loc[last_n_dates]
print(f"Using {len(last_n_dates)} trading dates: {last_n_dates[0].date()} to {last_n_dates[-1].date()}")

# ── keep only stocks with enough data ────────────────────────────────────────
valid_cols = close_n.columns[close_n.notna().sum() >= MIN_DAYS]
close_n    = close_n[valid_cols]
print(f"Stocks with >= {MIN_DAYS} days of data: {len(valid_cols)}")

close_n = close_n.ffill(limit=3)

# ── log returns ───────────────────────────────────────────────────────────────
returns = np.log(close_n / close_n.shift(1)).iloc[1:]
print(f"Returns matrix: {returns.shape}")

# ── correlation matrix (unique symbols only) ──────────────────────────────────
min_p = max(20, int(TARGET_DAYS * 0.4))
corr  = returns.corr(min_periods=min_p)

available_tickers = list(corr.columns)          # e.g. "TCS.NS"
available_syms    = [t.replace(".NS", "") for t in available_tickers]
available_sym_set = set(available_syms)
N                 = len(available_syms)
print(f"Correlation matrix: {N} x {N}")

# ── build symbols_meta: one entry per (symbol, industry) pair ────────────────
#    Only include rows where the symbol has price data
sym_to_name = {}
for _, row in df.iterrows():
    sym_to_name[row["Symbol"]] = row["Company Name"]   # last name wins (fine)

# Expand: for each available symbol, emit one entry per industry it belongs to
sym_industry_rows = df[df["Symbol"].isin(available_sym_set)].copy()
# sort so index aligns with the corr matrix order for the primary entry
sym_industry_rows = sym_industry_rows.sort_values(["Symbol","Industry"])

symbols_meta = []
for _, row in sym_industry_rows.iterrows():
    symbols_meta.append({
        "symbol":   row["Symbol"],
        "name":     row["Company Name"],
        "industry": row["Industry"],
    })

# ── unique industry list ──────────────────────────────────────────────────────
ind_list = sorted({m["industry"] for m in symbols_meta})

# ── build a sym->index map for the corr matrix ───────────────────────────────
sym_to_corr_idx = {sym: i for i, sym in enumerate(available_syms)}

# ── block model parameters ────────────────────────────────────────────────────
# Use the UNIQUE symbol corr matrix; industry grouping uses primary assignment
# (first industry alphabetically per symbol)
primary_industry = {}
for m in symbols_meta:
    sym = m["symbol"]
    if sym not in primary_industry:
        primary_industry[sym] = m["industry"]

corr_arr   = corr.values.astype(float)
rho_within = {}
rho_cnt    = {}
rho0_sum   = rho0_cnt = 0

for i in range(N):
    for j in range(i + 1, N):
        v = corr_arr[i, j]
        if np.isnan(v):
            continue
        ind_i = primary_industry.get(available_syms[i], "Unknown")
        ind_j = primary_industry.get(available_syms[j], "Unknown")
        if ind_i == ind_j:
            k             = ind_i
            rho_within[k] = rho_within.get(k, 0.0) + v
            rho_cnt[k]    = rho_cnt.get(k, 0) + 1
        else:
            rho0_sum += v
            rho0_cnt += 1

rho_within_avg = {k: rho_within[k] / rho_cnt[k] for k in rho_within if rho_cnt[k] > 0}
rho0      = rho0_sum / rho0_cnt if rho0_cnt > 0 else 0.0
all_off   = [corr_arr[i, j] for i in range(N) for j in range(i+1, N) if not np.isnan(corr_arr[i, j])]
rho_grand = float(np.mean(all_off)) if all_off else 0.0

# ── serialise ─────────────────────────────────────────────────────────────────
corr_list = [
    [None if np.isnan(v) else round(float(v), 5) for v in row]
    for row in corr_arr
]

out = {
    "generated_at":  datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "date_from":     str(last_n_dates[0].date()),
    "date_to":       str(last_n_dates[-1].date()),
    "trading_days":  TARGET_DAYS,
    "n_stocks":      N,
    "rho_grand":     round(rho_grand, 6),
    "rho0":          round(rho0, 6),
    "rho_within":    {k: round(v, 6) for k, v in rho_within_avg.items()},
    "industry_list": ind_list,
    "symbols":       symbols_meta,
    "corr_matrix":   corr_list,
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(out, f, separators=(",", ":"))

size_kb = os.path.getsize(OUT_PATH) / 1024
print(f"Saved {OUT_PATH} ({size_kb:.0f} KB)")
print(f"Stocks  : {N} unique symbols, {len(symbols_meta)} symbol-industry rows")
print(f"Days    : {TARGET_DAYS} ({out['date_from']} to {out['date_to']})")
print(f"rho_grand={rho_grand:.4f}  rho0={rho0:.4f}")
