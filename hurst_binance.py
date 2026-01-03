import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from binance_data import (
    load_cached_klines_range,
    fetch_top_market_cap_symbols,
)

CONFIG_PATH = os.getenv("HURST_CONFIG", "config.json")
SHOW_PLOTS = os.getenv("HURST_SHOW_PLOTS", "0") == "1"
PLOT_DIR = os.getenv("HURST_PLOT_DIR", "plot")
LOG_ENABLED = os.getenv("HURST_VERBOSE", "1") == "1"

DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
]
DEFAULT_NUM_CRYPTOS = len(DEFAULT_SYMBOLS)
QUOTE_ASSET = "USDT"
DEFAULT_DATA_DIR = "data"
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_INTERVAL = "1d"
DEFAULT_MIN_HISTORY_DAYS = 365


def load_config(path):
    config = {
        "num_cryptos": DEFAULT_NUM_CRYPTOS,
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

    num_cryptos = user_config.get("num_cryptos")
    if isinstance(num_cryptos, int) and num_cryptos > 0:
        config["num_cryptos"] = num_cryptos

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


# --- 1. FONCTION DE CALCUL DE L'EXPOSANT DE HURST ---
# C'est la fonction qui remplace le module caché "hurst_analysis" du YouTuber.
def calculate_hurst(ts):
    """
    Calcule l'exposant de Hurst d'une serie temporelle (Time Series)
    en utilisant la methode R/S (Rescaled Range Analysis).
    """
    ts = np.array(ts)
    if len(ts) < 100:  # Pas assez de donnees
        return np.nan

    # Creation des differentes echelles (lags)
    lags = range(2, 100)

    # Calcul des variances des differences (tau)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Regression lineaire sur les logs pour trouver la pente (H)
    # log(tau) = H * log(lag) + c
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    return poly[0] * 2.0  # *2 pour ajuster a la methode standard prix vs returns


# --- 2. LISTE DES PLUS GROS CRYPTOS (PAIRES USDT) ---
config = load_config(CONFIG_PATH)
log(f"Config loaded from {CONFIG_PATH}: {config}")
base_symbols = config["symbols"]
if not base_symbols:
    try:
        base_symbols = fetch_top_market_cap_symbols(
            config["num_cryptos"],
            quote_asset=QUOTE_ASSET,
            exclude_stablecoins=True,
        )
        log(f"Dynamic symbols selected: {base_symbols}")
    except Exception as exc:
        print(f"Auto market cap detection failed: {exc}")
        base_symbols = []
if not base_symbols:
    base_symbols = DEFAULT_SYMBOLS
    log(f"Fallback to default symbols: {base_symbols}")
num_cryptos = min(config["num_cryptos"], len(base_symbols))
symbols = base_symbols[:num_cryptos]
data_dir = config["data_dir"]
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


# --- 3. TELECHARGEMENT DES DONNEES ---

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
            close_series[symbol] = df["close"].dropna()
        else:
            log(f"Skipping {symbol}: no cached data in range.")
    except Exception as exc:
        print(f"Error for {symbol}: {exc}")

data = pd.DataFrame(close_series)


# --- 4. CALCUL DE L'EXPOSANT DE HURST POUR CHAQUE ACTIF ---
hurst_results = {}

print("Calculating Hurst exponent...")
for symbol in symbols:
    try:
        ts = data[symbol].dropna()
        if len(ts) > 100:
            h = calculate_hurst(ts.values)
            hurst_results[symbol] = h
    except Exception:
        continue

df_hurst = pd.DataFrame(list(hurst_results.items()), columns=["Ticker", "Hurst"])
df_hurst = df_hurst.dropna()


# --- 5. VISUALISATION ---
mean_reversion_threshold = 0.45
trending_threshold = 0.55

plt.figure(figsize=(12, 6))
plt.hist(df_hurst["Hurst"], bins=20, color="skyblue", edgecolor="black", alpha=0.7)
plt.axvline(0.5, color="red", linestyle="--", label="Random Walk (0.5)")
plt.axvline(trending_threshold, color="green", linestyle=":", label="Trending (>0.55)")
plt.axvline(mean_reversion_threshold, color="orange", linestyle=":", label="Range-Bound (<0.45)")
plt.title("Hurst Exponent Distribution (Binance Cryptos)")
plt.xlabel("Hurst Exponent")
plt.ylabel("Number of Assets")
plt.legend()
plt.grid(True, alpha=0.3)
finalize_plot("hurst_distribution.png")

if not df_hurst.empty:
    df_sorted = df_hurst.sort_values("Hurst", ascending=False)
    fig_height = max(6, 0.35 * len(df_sorted))
    plt.figure(figsize=(12, fig_height))
    bars = plt.barh(df_sorted["Ticker"], df_sorted["Hurst"], color="steelblue")
    plt.axvline(0.5, color="red", linestyle="--", label="Random Walk (0.5)")
    plt.axvline(trending_threshold, color="green", linestyle=":", label="Trending (>0.55)")
    plt.axvline(mean_reversion_threshold, color="orange", linestyle=":", label="Range (<0.45)")
    plt.gca().invert_yaxis()
    for bar in bars:
        width = bar.get_width()
        y_pos = bar.get_y() + bar.get_height() / 2
        plt.text(width + 0.01, y_pos, f"{width:.2f}", va="center", fontsize=8)
    x_min = min(0.0, df_sorted["Hurst"].min() - 0.05)
    x_max = df_sorted["Hurst"].max() + 0.1
    plt.xlim(x_min, x_max)
    plt.title("Hurst Exponent by Asset")
    plt.xlabel("Hurst Exponent")
    plt.ylabel("Asset")
    plt.legend()
    plt.grid(True, axis="x", alpha=0.3)
    finalize_plot("hurst_values.png")


# --- 6. AFFICHAGE DES RESULTATS ---
trending_assets = df_hurst[df_hurst["Hurst"] > trending_threshold].sort_values(
    by="Hurst", ascending=False
)
trending_count = min(num_cryptos, len(trending_assets))
print(f"\n--- TOP {trending_count} TRENDING ASSETS (H > 0.55) ---")
print(trending_assets.head(trending_count))

ranging_assets = df_hurst[df_hurst["Hurst"] < mean_reversion_threshold].sort_values(
    by="Hurst", ascending=True
)
ranging_count = min(num_cryptos, len(ranging_assets))
print(f"\n--- TOP {ranging_count} RANGE-BOUND ASSETS (H < 0.45) ---")
print(ranging_assets.head(ranging_count))


# --- 7. EXEMPLES GRAPHIQUES (TENDANCE + RANGE) ---
if not trending_assets.empty:
    top_ticker = trending_assets.iloc[0]["Ticker"]
    plt.figure(figsize=(12, 4))
    plt.plot(
        data[top_ticker],
        label=f"{top_ticker} (H={trending_assets.iloc[0]['Hurst']:.2f})",
    )
    plt.title(f"Most Trend-Following Asset: {top_ticker}")
    plt.legend()
    plt.grid(True)
    finalize_plot(f"{top_ticker}_trend.png")

if not ranging_assets.empty:
    low_ticker = ranging_assets.iloc[0]["Ticker"]
    plt.figure(figsize=(12, 4))
    plt.plot(
        data[low_ticker],
        label=f"{low_ticker} (H={ranging_assets.iloc[0]['Hurst']:.2f})",
    )
    plt.title(f"Most Range-Bound Asset: {low_ticker}")
    plt.legend()
    plt.grid(True)
    finalize_plot(f"{low_ticker}_range.png")
