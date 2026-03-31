import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "ind_nifty500list (1).csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "corr_data.json")

TARGET_DAYS = 252          # 1 full trading year
MIN_DAYS    = 200          # minimum days a stock must have to be included
FETCH_DAYS  = 540          # fetch ~2.1 calendar years to guarantee 252 trading days

# ── load stock list ────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
df["Symbol"]       = df["Symbol"].str.strip()
df["Industry"]     = df["Industry"].str.strip()
df["Company Name"] = df["Company Name"].str.strip()

stocks   = df[["Company Name","Industry","Symbol"]].dropna().to_dict("records")
tickers  = [s["Symbol"] + ".NS" for s in stocks]
print(f"Loaded {len(stocks)} stocks from CSV")

# ── fetch prices ───────────────────────────────────────────────────────────
end   = datetime.today()
start = end - timedelta(days=FETCH_DAYS)

print(f"Downloading {len(tickers)} tickers  ({start.date()} → {end.date()}) ...")
raw = yf.download(
    tickers,
    start=start.strftime("%Y-%m-%d"),
    end=end.strftime("%Y-%m-%d"),
    interval="1d",
    auto_adjust=True,
    progress=False,
    threads=True,
)

# ── extract Close panel ───────────────────────────────────────────────────
close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]

print(f"Raw price shape: {close.shape}")

# ── identify the 252 most-recent trading dates that NSE was open ──────────
# Use the date index from the broadest stock (most data points available)
# NSE trading calendar = any day where at least 10 stocks traded
trading_dates = close.index[close.notna().sum(axis=1) >= 10]

if len(trading_dates) < TARGET_DAYS:
    raise ValueError(f"Only {len(trading_dates)} common trading dates found — need {TARGET_DAYS}")

# Take the last 252 trading dates
last_252_dates = trading_dates[-TARGET_DAYS:]
close_252      = close.loc[last_252_dates]

print(f"Using {len(last_252_dates)} trading dates: {last_252_dates[0].date()} → {last_252_dates[-1].date()}")

# ── keep only stocks with enough data in this window ─────────────────────
valid_cols = close_252.columns[close_252.notna().sum() >= MIN_DAYS]
close_252  = close_252[valid_cols]
print(f"Stocks with >= {MIN_DAYS} days of data: {len(valid_cols)}")

# ── forward-fill up to 3 days (handle occasional trading halts) ──────────
close_252 = close_252.ffill(limit=3)

# ── compute log returns ───────────────────────────────────────────────────
returns = np.log(close_252 / close_252.shift(1)).iloc[1:]  # drop first NaN row
print(f"Returns matrix: {returns.shape}")

# ── correlation matrix (pairwise, NaN-safe) ───────────────────────────────
# Use pandas .corr() which handles per-pair NaNs automatically
corr = returns.corr(min_periods=100)

available_tickers = list(corr.columns)          # e.g. "TCS.NS"
available_syms    = [t.replace(".NS","") for t in available_tickers]
N = len(available_syms)
print(f"Correlation matrix: {N} x {N}")

# ── stock metadata ────────────────────────────────────────────────────────
sym_to_meta = {s["Symbol"]: s for s in stocks}
symbols_meta = []
for sym in available_syms:
    m = sym_to_meta.get(sym, {"Company Name": sym, "Industry": "Unknown", "Symbol": sym})
    symbols_meta.append({"symbol": sym, "name": m["Company Name"], "industry": m["Industry"]})

industries = [m["industry"] for m in symbols_meta]
ind_list   = sorted(set(industries))

# ── block model parameters ────────────────────────────────────────────────
corr_arr = corr.values.astype(float)

rho_within, rho_cnt = {}, {}
rho0_sum = rho0_cnt = 0

for i in range(N):
    for j in range(i + 1, N):
        v = corr_arr[i, j]
        if np.isnan(v):
            continue
        if industries[i] == industries[j]:
            k = industries[i]
            rho_within[k] = rho_within.get(k, 0.0) + v
            rho_cnt[k]    = rho_cnt.get(k, 0) + 1
        else:
            rho0_sum += v
            rho0_cnt += 1

rho_within_avg = {k: rho_within[k]/rho_cnt[k] for k in rho_within if rho_cnt[k] > 0}
rho0           = rho0_sum / rho0_cnt if rho0_cnt > 0 else 0.0

all_off   = [corr_arr[i,j] for i in range(N) for j in range(i+1,N) if not np.isnan(corr_arr[i,j])]
rho_grand = float(np.mean(all_off)) if all_off else 0.0

# ── serialise ─────────────────────────────────────────────────────────────
corr_list = [[None if np.isnan(v) else round(float(v),5) for v in row] for row in corr_arr]

out = {
    "generated_at":  datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "date_from":     str(last_252_dates[0].date()),
    "date_to":       str(last_252_dates[-1].date()),
    "trading_days":  TARGET_DAYS,
    "n_stocks":      N,
    "rho_grand":     round(rho_grand,  6),
    "rho0":          round(rho0,       6),
    "rho_within":    {k: round(v,6) for k,v in rho_within_avg.items()},
    "industry_list": ind_list,
    "symbols":       symbols_meta,
    "corr_matrix":   corr_list,
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(out, f, separators=(",",":"))

size_kb = os.path.getsize(OUT_PATH)/1024
print(f"\nSaved  {OUT_PATH}  ({size_kb:.0f} KB)")
print(f"Stocks : {N}")
print(f"Days   : {TARGET_DAYS}  ({out['date_from']} → {out['date_to']})")
print(f"rho_grand={rho_grand:.4f}   rho0={rho0:.4f}")
