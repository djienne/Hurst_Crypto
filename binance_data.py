from datetime import datetime, timezone
import asyncio
import json
import os
import threading
import time

import pandas as pd
import requests

BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
BINANCE_PRODUCTS_URL = "https://www.binance.com/bapi/asset/v2/public/asset-service/product/get-products"
MAX_KLINES_PER_REQUEST = 1000
LOG_ENABLED = os.getenv("HURST_VERBOSE", "1") == "1"
CONFIG_PATH = os.getenv("HURST_CONFIG", "config.json")
DEFAULT_RATE_LIMIT_SECONDS = 0.2
RATE_LIMIT_SECONDS = DEFAULT_RATE_LIMIT_SECONDS
_LAST_REQUEST_AT = 0.0
_RATE_LIMIT_LOCK = threading.Lock()
STABLECOIN_TAGS = {"stablecoin"}
STABLECOIN_BASES = {
    "USDT",
    "USDC",
    "BUSD",
    "TUSD",
    "USDP",
    "DAI",
    "FDUSD",
    "PYUSD",
    "GUSD",
    "FRAX",
    "LUSD",
    "EURS",
    "EURT",
    "USDD",
    "USTC",
    "UST",
}
INTERVAL_FREQUENCY_MAP = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "8h": "8H",
    "12h": "12H",
    "1d": "1D",
    "3d": "3D",
    "1w": "1W",
}


def _log(message):
    if LOG_ENABLED:
        print(message)


def _warn(message):
    print(message)


def _load_config(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    if not isinstance(config, dict):
        return {}
    return config


_CONFIG = _load_config(CONFIG_PATH)
_config_rate_limit = _CONFIG.get("rate_limit_seconds")
if isinstance(_config_rate_limit, (int, float)) and _config_rate_limit >= 0:
    RATE_LIMIT_SECONDS = float(_config_rate_limit)


def _rate_limited_get(url, **kwargs):
    global _LAST_REQUEST_AT
    delay = 0.0
    if RATE_LIMIT_SECONDS > 0:
        with _RATE_LIMIT_LOCK:
            now = time.monotonic()
            elapsed = now - _LAST_REQUEST_AT
            if elapsed < RATE_LIMIT_SECONDS:
                delay = RATE_LIMIT_SECONDS - elapsed
            _LAST_REQUEST_AT = now + delay
    if delay > 0:
        _log(f"Rate limiting: sleeping {delay:.2f}s")
        time.sleep(delay)
    return requests.get(url, **kwargs)


def _to_millis(dt):
    dt = _to_datetime(dt)
    return int(dt.timestamp() * 1000)


def _to_datetime(value):
    if isinstance(value, pd.Timestamp):
        dt = value.to_pydatetime()
    elif isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value)
    else:
        raise TypeError("Unsupported datetime value")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def fetch_binance_klines(symbol, interval, start, end):
    start_ms = _to_millis(start)
    end_ms = _to_millis(end) if end else None
    rows = []

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "limit": MAX_KLINES_PER_REQUEST,
        }
        if end_ms is not None:
            params["endTime"] = end_ms

        response = _rate_limited_get(BINANCE_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not data:
            break

        rows.extend(data)

        last_open_time = data[-1][0]
        next_start_ms = last_open_time + 1
        if next_start_ms <= start_ms:
            break

        start_ms = next_start_ms
        if end_ms is not None and start_ms >= end_ms:
            break
        if len(data) < MAX_KLINES_PER_REQUEST:
            break

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_stablecoin(item):
    tags = item.get("tags") or []
    base = item.get("b")
    if any(tag in STABLECOIN_TAGS for tag in tags):
        return True
    if base and base.upper() in STABLECOIN_BASES:
        return True
    return False


def fetch_top_market_cap_symbols(num_cryptos, quote_asset="USDT", exclude_stablecoins=True):
    _log(f"Fetching Binance products to rank top {num_cryptos} by market cap...")
    response = _rate_limited_get(
        BINANCE_PRODUCTS_URL,
        params={"include_etf": "true"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data")

    if isinstance(data, dict):
        rows = data.get("rows") or data.get("list") or data.get("data") or []
    else:
        rows = data or []
    _log(f"Binance products received: {len(rows)} entries")

    candidates = []
    for item in rows:
        if item.get("q") != quote_asset:
            continue
        if exclude_stablecoins and _is_stablecoin(item):
            continue

        price = _to_float(item.get("c"))
        supply = _to_float(item.get("cs"))
        market_cap = price * supply
        if market_cap <= 0:
            continue

        symbol = item.get("s")
        if symbol:
            candidates.append((symbol, market_cap))

    candidates.sort(key=lambda entry: entry[1], reverse=True)
    selected = [symbol for symbol, _ in candidates[:num_cryptos]]
    _log(f"Selected symbols: {selected}")
    return selected


def _interval_to_frequency(interval):
    return INTERVAL_FREQUENCY_MAP.get(interval)


def _normalize_klines(df, label=None):
    if df.empty:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    if df.index.isna().any():
        if label:
            print(f"Dropping NaT rows for {label}")
        df = df.loc[~df.index.isna()]

    needs_sort = not df.index.is_monotonic_increasing
    has_dupes = df.index.has_duplicates
    if needs_sort or has_dupes:
        if label:
            print(f"Reordering cached data for {label}")
        df = df.sort_index()
        if has_dupes:
            df = df.loc[~df.index.duplicated(keep="last")]

    return df


def _cache_path(symbol, data_dir):
    return os.path.join(data_dir, f"{symbol}.csv")


def load_cached_klines(symbol, data_dir):
    path = _cache_path(symbol, data_dir)
    if not os.path.exists(path):
        _log(f"Cache miss: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
        df.set_index("open_time", inplace=True)
    else:
        fallback_col = df.columns[0]
        df[fallback_col] = pd.to_datetime(df[fallback_col], utc=True, errors="coerce")
        df.set_index(fallback_col, inplace=True)
        df.index.name = "open_time"
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = _normalize_klines(df, label=symbol)
    _log(f"Cache hit: {path} ({len(df)} rows)")
    return df


def save_cached_klines(symbol, df, data_dir):
    if df.empty:
        return

    os.makedirs(data_dir, exist_ok=True)
    path = _cache_path(symbol, data_dir)
    df.to_csv(path, index_label="open_time")
    _log(f"Cache saved: {path} ({len(df)} rows)")


def _expected_index(start_dt, end_dt, interval):
    freq = _interval_to_frequency(interval)
    if not freq:
        return None

    start_ts = pd.Timestamp(start_dt)
    end_ts = pd.Timestamp(end_dt)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")

    return pd.date_range(start=start_ts, end=end_ts, freq=freq)


def _collapse_missing_ranges(missing_index, step):
    if missing_index.empty:
        return []

    missing_sorted = missing_index.sort_values()
    ranges = []
    start = missing_sorted[0]
    prev = missing_sorted[0]

    for ts in missing_sorted[1:]:
        if step is not None and ts - prev == step:
            prev = ts
            continue
        ranges.append((start, prev))
        start = prev = ts

    ranges.append((start, prev))
    return ranges


def _count_candles(start, end, step):
    if step is None:
        return None
    return int((end - start) / step) + 1


def _format_gap(start, end, count):
    if count is None:
        return f"{start} -> {end}"
    return f"{start} -> {end} ({count} candles)"


def _report_gaps(symbol, expected_index, actual_index):
    if expected_index is None or expected_index.empty:
        return

    missing = expected_index.difference(actual_index)
    if missing.empty:
        _log(f"{symbol}: no gaps detected.")
        return

    expected_start = expected_index[0]
    expected_end = expected_index[-1]
    missing_count = len(missing)
    if missing_count == len(expected_index):
        _warn(
            f"Warning: {symbol} missing entire range "
            f"{expected_start} -> {expected_end} ({missing_count} candles)"
        )
        return

    step = expected_index[1] - expected_index[0] if len(expected_index) > 1 else None
    ranges = _collapse_missing_ranges(missing, step)

    leading = []
    trailing = []
    middle = []
    for start, end in ranges:
        if start == expected_start:
            leading.append((start, end))
            continue
        if end == expected_end:
            trailing.append((start, end))
            continue
        middle.append((start, end))

    if leading:
        for start, end in leading:
            count = _count_candles(start, end, step)
            _warn(f"Warning: {symbol} leading gap {_format_gap(start, end, count)}")
    if middle:
        max_show = 5
        for start, end in middle[:max_show]:
            count = _count_candles(start, end, step)
            _warn(f"Warning: {symbol} middle gap {_format_gap(start, end, count)}")
        if len(middle) > max_show:
            _warn(
                f"Warning: {symbol} {len(middle) - max_show} additional middle gaps not shown."
            )
    if trailing:
        for start, end in trailing:
            count = _count_candles(start, end, step)
            _warn(f"Warning: {symbol} trailing gap {_format_gap(start, end, count)}")


def fetch_binance_klines_cached(symbol, interval, start, end, data_dir="data"):
    start_dt = _to_datetime(start)
    end_dt = _to_datetime(end) if end else datetime.now(timezone.utc)

    _log(
        f"Fetching {symbol} {interval} from {start_dt.isoformat()} to {end_dt.isoformat()}"
    )
    cached_full = load_cached_klines(symbol, data_dir)

    def _safe_slice(df):
        if df.empty:
            return df
        try:
            return df.loc[start_dt:end_dt]
        except KeyError:
            df = _normalize_klines(df, label=symbol)
            return df.loc[start_dt:end_dt]

    if cached_full.empty:
        cached_range = cached_full
    else:
        cached_range = _safe_slice(cached_full)

    expected = _expected_index(start_dt, end_dt, interval)
    if expected is None:
        missing_ranges = []
        if cached_range.empty:
            missing_ranges.append((start_dt, end_dt))
    else:
        missing = expected.difference(cached_range.index)
        if missing.empty:
            missing_ranges = []
        else:
            step = expected[1] - expected[0] if len(expected) > 1 else None
            missing_ranges = _collapse_missing_ranges(missing, step)

    _log(
        f"{symbol} cached rows: {len(cached_range)} missing ranges: {len(missing_ranges)}"
    )
    if missing_ranges:
        _log(f"{symbol} attempting to fill {len(missing_ranges)} gap range(s).")
    else:
        _log(f"{symbol} no gaps to fill.")
    new_frames = []
    for missing_start, missing_end in missing_ranges:
        _log(
            f"{symbol} downloading range {missing_start.isoformat()} -> {missing_end.isoformat()}"
        )
        new_frames.append(fetch_binance_klines(symbol, interval, missing_start, missing_end))

    if new_frames:
        merged = pd.concat([cached_full] + new_frames, axis=0)
    else:
        merged = cached_full

    merged = _normalize_klines(merged, label=symbol)
    save_cached_klines(symbol, merged, data_dir)

    if expected is not None:
        merged_range = merged.loc[start_dt:end_dt]
        _report_gaps(symbol, expected, merged_range.index)

    return merged.loc[start_dt:end_dt]


def load_cached_klines_range(symbol, interval, start, end, data_dir="data"):
    start_dt = _to_datetime(start)
    end_dt = _to_datetime(end) if end else datetime.now(timezone.utc)

    _log(
        f"Loading cached {symbol} {interval} from {start_dt.isoformat()} to {end_dt.isoformat()}"
    )
    cached_full = load_cached_klines(symbol, data_dir)
    if cached_full.empty:
        _log(f"No cached data found for {symbol}.")
        return cached_full

    def _safe_slice(df):
        if df.empty:
            return df
        try:
            return df.loc[start_dt:end_dt]
        except KeyError:
            df = _normalize_klines(df, label=symbol)
            return df.loc[start_dt:end_dt]

    cached_range = _safe_slice(cached_full)
    if cached_range.empty:
        _log(f"No cached data for {symbol} in the requested range.")
        return cached_range

    expected = _expected_index(start_dt, end_dt, interval)
    _report_gaps(symbol, expected, cached_range.index)

    return cached_range


async def _download_symbol_async(symbol, interval, start, end, data_dir, semaphore):
    async with semaphore:
        _log(f"Downloading data for {symbol}...")
        rows = await asyncio.to_thread(
            _download_symbol_sync,
            symbol,
            interval,
            start,
            end,
            data_dir,
        )
        return symbol, rows


def _download_symbol_sync(symbol, interval, start, end, data_dir):
    df = fetch_binance_klines_cached(symbol, interval, start, end, data_dir)
    return len(df)


async def download_symbols_async(symbols, interval, start, end, data_dir, max_concurrency):
    if not symbols:
        return []
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        asyncio.create_task(
            _download_symbol_async(symbol, interval, start, end, data_dir, semaphore)
        )
        for symbol in symbols
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results


if __name__ == "__main__":
    import argparse

    config = _load_config(CONFIG_PATH)
    default_start = "2018-01-01"
    config_start = config.get("start_date")
    if isinstance(config_start, str) and config_start:
        default_start = config_start
    config_top = config.get("num_cryptos")
    default_top = None
    default_interval = "1d"
    config_interval = config.get("interval")
    if isinstance(config_interval, str) and config_interval:
        default_interval = config_interval
    default_data_dir = "data"
    config_data_dir = config.get("data_dir")
    if isinstance(config_data_dir, str) and config_data_dir:
        default_data_dir = config_data_dir
    default_concurrency = 5
    config_concurrency = config.get("download_concurrency")
    if isinstance(config_concurrency, int) and config_concurrency > 0:
        default_concurrency = config_concurrency

    parser = argparse.ArgumentParser(
        description="Binance data utilities (market cap ranking + cached klines)."
    )
    parser.add_argument(
        "--top",
        type=int,
        default=default_top,
        help="Number of symbols to list.",
    )
    parser.add_argument("--quote", default="USDT", help="Quote asset filter (default: USDT).")
    parser.add_argument(
        "--include-stablecoins",
        action="store_true",
        help="Include stablecoins in the ranking.",
    )
    parser.add_argument(
        "--sample-symbol",
        default="",
        help="Fetch cached klines for a sample symbol (e.g., BTCUSDT).",
    )
    parser.add_argument(
        "--interval",
        default=default_interval,
        help="Kline interval (default: 1d).",
    )
    parser.add_argument("--start", default=default_start, help="Start date (ISO).")
    parser.add_argument("--end", default=None, help="End date (ISO).")
    parser.add_argument("--data-dir", default=default_data_dir, help="Cache directory.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=default_concurrency,
        help="Max parallel downloads (default from config).",
    )
    args = parser.parse_args()

    if args.top is not None:
        top = args.top
    elif isinstance(config_top, int) and config_top > 0:
        top = config_top
    else:
        top = 10

    _log(f"Resolved top count: {top}")
    symbols = fetch_top_market_cap_symbols(
        top,
        quote_asset=args.quote,
        exclude_stablecoins=not args.include_stablecoins,
    )
    print(f"Top symbols ({len(symbols)}): {symbols}")

    if args.sample_symbol:
        symbols_to_download = [args.sample_symbol]
    else:
        symbols_to_download = symbols

    print(
        f"Downloading data for {len(symbols_to_download)} symbols "
        f"(concurrency={args.concurrency})..."
    )
    results = asyncio.run(
        download_symbols_async(
            symbols_to_download,
            args.interval,
            args.start,
            args.end,
            args.data_dir,
            args.concurrency,
        )
    )
    for result in results:
        if isinstance(result, Exception):
            print(f"Download error: {result}")
            continue
        symbol, rows = result
        print(f"{symbol} rows: {rows}")
