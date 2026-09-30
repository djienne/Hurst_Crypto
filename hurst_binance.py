import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from binance_data import load_cached_klines_range
from hurst import LAGS, null_test, verdict

CONFIG_PATH = os.getenv("HURST_CONFIG", "config.json")
SHOW_PLOTS = os.getenv("HURST_SHOW_PLOTS", "0") == "1"
PLOT_DIR = os.getenv("HURST_PLOT_DIR", "plot")
LOG_ENABLED = os.getenv("HURST_VERBOSE", "1") == "1"

DEFAULT_DATA_DIR = "data"
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_INTERVAL = "1d"
DEFAULT_MIN_HISTORY_DAYS = 365
ALPHA = 0.05
# Monte-Carlo error on p is ~2*sqrt(p(1-p)/N_SHUFFLE): +-0.01 near p=0.05 with 2000;
# 300 shuffles (+-0.025) flipped borderline verdicts between seeds.
N_SHUFFLE = 2000


def load_config(path):
    config = {
        "symbols": None,
        "data_dir": DEFAULT_DATA_DIR,
        "start_date": DEFAULT_START_DATE,
        "interval": DEFAULT_INTERVAL,
        "min_history_days": DEFAULT_MIN_HISTORY_DAYS,
    }
    try:
        with open(path, "r", encoding="utf-8") as handle:
            user_config = json.load(handle)
    except FileNotFoundError:
        if LOG_ENABLED:
            print(f"Config file missing: {path}. Using defaults.")
        return config
    except json.JSONDecodeError:
        print(f"Invalid config JSON: {path}. Using defaults.")
        return config

    if not isinstance(user_config, dict):
        return config

    symbols = user_config.get("symbols")
    if isinstance(symbols, list) and all(isinstance(item, str) for item in symbols):
        config["symbols"] = symbols

    data_dir = user_config.get("data_dir")
    if isinstance(data_dir, str) and data_dir:
        config["data_dir"] = data_dir

    start_date = user_config.get("start_date")
    if isinstance(start_date, str) and start_date:
        config["start_date"] = start_date

    interval = user_config.get("interval")
    if isinstance(interval, str) and interval:
        config["interval"] = interval

    min_history_days = user_config.get("min_history_days")
    if isinstance(min_history_days, int) and min_history_days > 0:
        config["min_history_days"] = min_history_days

    return config


def log(message):
    if LOG_ENABLED:
        print(message)


def finalize_plot(filename):
    if SHOW_PLOTS:
        plt.show()
        return
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()


# --- 1. UNIVERSE: config "symbols", else every cached symbol (no network) ---
config = load_config(CONFIG_PATH)
log(f"Config loaded from {CONFIG_PATH}: {config}")
data_dir = config["data_dir"]
symbols = config["symbols"] or sorted(
    name[:-4] for name in os.listdir(data_dir) if name.endswith(".csv")
)
start_date = config["start_date"]
interval = config["interval"]
min_history_days = config["min_history_days"]
end_date = None
end_date_label = "now"
log(f"Using {len(symbols)} symbols: {symbols}")
log(f"Cache directory: {data_dir}")
log(f"Plots directory: {PLOT_DIR} (show={SHOW_PLOTS})")
log(f"Date range: {start_date} -> {end_date_label}")
log(f"Interval: {interval}")
log(f"Minimum history: {min_history_days} days")
log("Cache-only mode: no downloads will be performed.")


# --- 2. LOAD CACHED CLOSES ---
print(f"Loading cached Binance data for {len(symbols)} cryptos...")
close_series = {}
for symbol in symbols:
    try:
        df = load_cached_klines_range(symbol, interval, start_date, end_date, data_dir)
        if not df.empty:
            history_span = df.index.max() - df.index.min()
            if history_span < pd.Timedelta(days=min_history_days):
                log(
                    f"Skipping {symbol}: history {history_span.days} days "
                    f"< {min_history_days} days."
                )
                continue
            close = df["close"].dropna()
            if (close <= 0).any():
                print(f"Skipping {symbol}: non-positive close prices.")
                continue
            close_series[symbol] = close
        else:
            log(f"Skipping {symbol}: no cached data in range.")
    except Exception as exc:
        print(f"Error for {symbol}: {exc}")


# --- 3. HURST EXPONENT OF LOG PRICE VS ITS SHUFFLED NULL ---
print(f"Calculating Hurst exponent ({N_SHUFFLE} shuffles per asset)...")
rows = []
for symbol, close in close_series.items():
    h, null_mean, null_sd, p = null_test(np.log(close.to_numpy()), N_SHUFFLE)
    rows.append({
        "Ticker": symbol,
        "N": len(close),
        "First": close.index[0].date(),
        "Last": close.index[-1].date(),
        "Hurst": h,
        "NullMean": null_mean,
        "NullSD": null_sd,
        "z": (h - null_mean) / null_sd,
        "p": p,
        "Verdict": verdict(h, null_mean, p, ALPHA),
    })
if not rows:
    raise SystemExit("No asset with enough cached history.")

df_hurst = pd.DataFrame(rows).dropna(subset=["Hurst"])
df_hurst = df_hurst.sort_values("Hurst", ascending=False).reset_index(drop=True)
print(df_hurst.round(3).to_string(index=False))
n_significant = (df_hurst["p"] < ALPHA).sum()
print(
    f"\n{n_significant} of {len(df_hurst)} assets differ from their shuffled null at p < {ALPHA} "
    f"(~{ALPHA * len(df_hurst):.1f} expected by chance; assets co-move, "
    f"so there are fewer independent tests than assets)."
)
os.makedirs(PLOT_DIR, exist_ok=True)
df_hurst.assign(
    Interval=interval, Lags=f"{LAGS[0]}-{LAGS[-1]}", Shuffles=N_SHUFFLE
).to_csv(os.path.join(PLOT_DIR, "hurst_results.csv"), index=False)


# --- 4. VISUALISATION ---
VERDICT_COLORS = {
    "persistent": "seagreen",
    "anti-persistent": "darkorange",
    "no memory detected": "lightgray",
}

plt.figure(figsize=(12, 6))
plt.hist(df_hurst["Hurst"] - df_hurst["NullMean"], bins=12, color="skyblue", edgecolor="black", alpha=0.7)
plt.axvline(0, color="red", linestyle="--", label="No memory (shuffled-null mean)")
plt.title("Hurst Exponent minus Shuffled-Null Mean (Binance Cryptos, log price)")
plt.xlabel("H - H_null")
plt.ylabel("Number of Assets")
plt.legend()
plt.grid(True, alpha=0.3)
finalize_plot("hurst_distribution.png")

y_pos = np.arange(len(df_hurst))
plt.figure(figsize=(12, max(6, 0.35 * len(df_hurst))))
for name, color in VERDICT_COLORS.items():
    mask = (df_hurst["Verdict"] == name).to_numpy()
    if not mask.any():
        continue
    label = name if name == "no memory detected" else f"{name} (p < {ALPHA})"
    plt.barh(y_pos[mask], df_hurst["Hurst"][mask], color=color, label=label)
plt.errorbar(
    df_hurst["NullMean"], y_pos, xerr=1.96 * df_hurst["NullSD"],
    fmt="|", color="black", capsize=3, label="Shuffled null (mean ± 1.96 sd)",
)
plt.axvline(0.5, color="red", linestyle=":", alpha=0.5, label="0.5 (random walk, N → ∞)")
null_high = df_hurst["NullMean"] + 1.96 * df_hurst["NullSD"]
for y, h, x_text in zip(y_pos, df_hurst["Hurst"], np.maximum(df_hurst["Hurst"], null_high)):
    plt.text(x_text + 0.01, y, f"{h:.2f}", va="center", fontsize=8)
plt.yticks(y_pos, df_hurst["Ticker"])
plt.gca().invert_yaxis()
plt.xlim(0.0, max(df_hurst["Hurst"].max(), null_high.max()) + 0.1)
plt.title("Hurst Exponent by Asset (log price) vs Shuffled Null")
plt.xlabel("Hurst Exponent")
plt.ylabel("Asset")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.grid(True, axis="x", alpha=0.3)
finalize_plot("hurst_values.png")


# --- 5. EXAMPLE CHARTS: MOST SIGNIFICANT ASSET ON EACH SIDE ---
for name, filename in (
    ("persistent", "example_persistent.png"),
    ("anti-persistent", "example_antipersistent.png"),
):
    subset = df_hurst[df_hurst["Verdict"] == name]
    if subset.empty:
        log(f"No {name} asset at p < {ALPHA}; no {filename}.")
        stale = os.path.join(PLOT_DIR, filename)
        if os.path.exists(stale):
            os.remove(stale)
        continue
    best = subset.loc[subset["z"].abs().idxmax()]
    plt.figure(figsize=(12, 4))
    plt.plot(
        close_series[best["Ticker"]],
        label=(
            f"{best['Ticker']} (H={best['Hurst']:.2f}, null {best['NullMean']:.2f}"
            f" ± {best['NullSD']:.2f}, p={best['p']:.3f})"
        ),
    )
    plt.yscale("log")
    plt.title(f"Most Significant {name.capitalize()} Asset: {best['Ticker']}")
    plt.ylabel("Close (USDT, log scale)")
    plt.legend()
    plt.grid(True)
    finalize_plot(filename)
