#!/usr/bin/env python3
# stock_lookup.py

import sys
import re
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf

# Optional name -> ticker search
try:
    from yahooquery import search as yq_search  # type: ignore
    HAVE_YQ = True
except Exception:
    HAVE_YQ = False


# Exchange code and name whitelists for US and CA
EXCHANGE_CODES_US = {"NMS", "NCM", "NGM", "NYQ", "ASE", "PCX", "BATS", "CBOE"}
EXCHANGE_NAMES_US = {"NASDAQ", "NYSE", "NYSE ARCA", "NYSE AMERICAN", "ARCA", "AMEX", "CBOE", "BATS"}

EXCHANGE_CODES_CA = {"TOR", "TSX", "TSXV", "V", "NEO", "CNQ", "CNSX"}
EXCHANGE_NAMES_CA = {"TORONTO", "TSX", "TSX VENTURE", "NEO", "CANADIAN SECURITIES", "CSE", "CBOE CANADA"}

CA_SUFFIXES = {"TO", "V", "NE", "CN"}

# Tight ticker regex so words like "nvidia" do not look like tickers
TICKER_RE = re.compile(r"^[A-Z0-9]{1,5}(\.(TO|V|NE|CN))?$")


def is_likely_ticker(q: str) -> bool:
    q = q.strip().upper()
    return bool(TICKER_RE.match(q))


def _exchange_is_us_ca(exch: Optional[str], currency: Optional[str] = None) -> bool:
    u = (exch or "").upper()
    if u in EXCHANGE_CODES_US or u in EXCHANGE_CODES_CA:
        return True
    names = EXCHANGE_NAMES_US | EXCHANGE_NAMES_CA
    if any(name in u for name in names):
        return True
    if currency and currency.upper() in {"USD", "CAD"}:
        return True
    return False


def _symbol_is_us_ca(sym: str) -> bool:
    s = sym.upper()
    if "." not in s:
        # No suffix means US on Yahoo for normal equities
        return 1 <= len(s) <= 5
    suf = s.rsplit(".", 1)[-1]
    return suf in CA_SUFFIXES


def resolve_symbol(query: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Returns (symbol, longname, exchangeName) for US or CA only.
    """
    q = query.strip()

    if is_likely_ticker(q):
        sym = q.upper()
        if not _symbol_is_us_ca(sym):
            raise ValueError(f"Only US or CA listings allowed. Try AAPL or RY.TO. Got '{sym}'.")
        return sym, None, None

    if not HAVE_YQ:
        raise ValueError("Company-name search needs 'yahooquery'. Install it or enter a ticker like NVDA or RY.TO.")

    res = yq_search(q)
    quotes = res.get("quotes", []) if isinstance(res, dict) else []
    if not quotes:
        raise ValueError("No matches. Enter a valid US or CA ticker like NVDA or RY.TO.")

    def good(item: Dict[str, Any]) -> bool:
        exch = (item.get("exchange") or item.get("fullExchangeName") or "").upper()
        sym = (item.get("symbol") or item.get("ticker") or "").upper()
        return _exchange_is_us_ca(exch) or _symbol_is_us_ca(sym)

    us_ca = [it for it in quotes if good(it)]
    if not us_ca:
        raise ValueError("Only US or CA listings are supported. Your query matched non-US/CA results.")

    def score(item: Dict[str, Any]) -> int:
        s = 0
        if item.get("quoteType") == "EQUITY":
            s += 5
        name_blob = f"{item.get('longname','')} {item.get('shortname','')}".lower()
        if q.lower() in name_blob:
            s += 2
        exch = (item.get("exchange") or item.get("fullExchangeName") or "")
        if _exchange_is_us_ca(exch):
            s += 2
        return s

    us_ca.sort(key=score, reverse=True)
    best = us_ca[0]
    symbol = (best.get("symbol") or best.get("ticker") or q).upper()
    longname = best.get("longname") or best.get("shortname")
    exchname = best.get("fullExchangeName") or best.get("exchange")
    return symbol, longname, exchname


def _pick_series(df_like: Any, symbol: str, field_name: str) -> pd.Series:
    """
    Given df[field_name] which can be Series or DataFrame with multiple tickers,
    pick a single 1D Series. Prefer the exact ticker if present, otherwise the
    column with the most data.
    """
    if isinstance(df_like, pd.Series):
        return pd.to_numeric(df_like, errors="coerce")

    if not isinstance(df_like, pd.DataFrame):
        raise ValueError(f"Unexpected type for {field_name}: {type(df_like)}")

    if df_like.shape[1] == 0:
        raise ValueError(f"No data in column '{field_name}'")

    cols = list(df_like.columns)
    # Try exact symbol match first
    if symbol in df_like.columns:
        s = df_like[symbol]
        return pd.to_numeric(s, errors="coerce")

    # Otherwise pick the densest column
    best_col = df_like.count().idxmax()
    s = df_like[best_col]
    return pd.to_numeric(s, errors="coerce")


def fetch_ohlcv(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df_raw = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if df_raw is None or df_raw.empty:
        raise ValueError(f"Could not fetch OHLCV for {symbol} with period={period} interval={interval}")

    # Normalize to single Series per field, even if yahoo returns multi-index or multi-ticker frames
    # df_raw['X'] might be Series or a DataFrame of tickers
    fields = {}
    for field in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if field in df_raw.columns:
            fields[field] = _pick_series(df_raw[field], symbol, field)
        else:
            # Sometimes Adj Close is missing intraday
            if field == "Adj Close" and "Close" in df_raw.columns:
                fields[field] = _pick_series(df_raw["Close"], symbol, "Adj Close")
            else:
                raise ValueError(f"Field '{field}' missing in downloaded data for {symbol}")

    out = pd.DataFrame(fields).sort_index()
    # Drop rows with everything NaN
    out = out.dropna(how="all")
    if out.empty:
        raise ValueError(f"No valid rows for {symbol} after normalization")
    return out


def get_key_info(symbol: str) -> Dict[str, Any]:
    t = yf.Ticker(symbol)
    info = {}
    try:
        info = t.get_info() or {}
    except Exception:
        pass
    fast = {}
    try:
        fast = dict(t.fast_info) if hasattr(t, "fast_info") else {}
    except Exception:
        pass
    merged = {**fast, **info}
    out = {
        "symbol": symbol.upper(),
        "shortName": merged.get("shortName") or merged.get("longName"),
        "currency": merged.get("currency") or merged.get("currencySymbol") or merged.get("financialCurrency"),
        "exchange": merged.get("exchange") or merged.get("fullExchangeName"),
        "marketCap": merged.get("marketCap"),
        "trailingPE": merged.get("trailingPE") or merged.get("peTrailing"),
        "forwardPE": merged.get("forwardPE") or merged.get("peForward"),
        "dividendYield": merged.get("dividendYield") or merged.get("trailingAnnualDividendYield"),
        "beta": merged.get("beta") or merged.get("beta3Year"),
        "fiftyTwoWeekLow": merged.get("fiftyTwoWeekLow"),
        "fiftyTwoWeekHigh": merged.get("fiftyTwoWeekHigh"),
        "avgVolume": merged.get("averageVolume") or merged.get("averageDailyVolume10Day") or merged.get("threeMonthAverageVolume"),
        "sharesOutstanding": merged.get("sharesOutstanding"),
        "sector": merged.get("sector"),
        "industry": merged.get("industry"),
        "website": merged.get("website"),
        "lastPrice": merged.get("lastPrice") or merged.get("regularMarketPrice") or merged.get("currentPrice"),
    }
    return out


def humanize_int(n: Optional[float]) -> Optional[str]:
    if n is None:
        return None
    n = float(n)
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000.0:
            return f"{n:,.0f}{unit}"
        n /= 1000.0
    return f"{n:.1f}P"


def compute_quick_stats(df: pd.DataFrame) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}

    # Align and drop NaN rows once, to keep shapes sane
    aligned = df[["High", "Low", "Adj Close", "Volume"]].copy().dropna()
    if aligned.empty:
        return {
            "period_return_pct": None,
            "ann_vol_pct": None,
            "sma_20": None,
            "sma_50": None,
            "sma_200": None,
            "avg_volume": None,
            "atr_14": None,
        }

    closes = aligned["Adj Close"]
    highs = aligned["High"]
    lows = aligned["Low"]
    vols = aligned["Volume"]

    # Convert to 1D numpy arrays
    cvals = closes.to_numpy(dtype=float)
    hvals = highs.to_numpy(dtype=float)
    lvals = lows.to_numpy(dtype=float)
    vvals = vols.to_numpy(dtype=float)

    # Period return
    if cvals.size >= 2 and np.isfinite(cvals[[0, -1]]).all():
        period_return = float(cvals[-1] / cvals[0] - 1.0)
        stats["period_return_pct"] = round(period_return * 100, 2)
    else:
        stats["period_return_pct"] = None

    # Annualized volatility from daily pct change
    if cvals.size > 2:
        rets = np.diff(cvals) / cvals[:-1]
        rets = rets[np.isfinite(rets)]
        stats["ann_vol_pct"] = round(float(rets.std(ddof=1) * np.sqrt(252.0)) * 100, 2) if rets.size > 2 else None
    else:
        stats["ann_vol_pct"] = None

    # SMAs from pandas to keep it simple
    for w in [20, 50, 200]:
        if len(closes) >= w:
            sma = float(closes.rolling(w).mean().iloc[-1])
            stats[f"sma_{w}"] = round(sma, 4)
        else:
            stats[f"sma_{w}"] = None

    # Average volume
    stats["avg_volume"] = int(np.nanmean(vvals)) if vvals.size else None

    # ATR(14)
    if len(closes) >= 15:
        prev_close = np.r_[np.nan, cvals[:-1]]
        tr = np.nanmax(
            np.c_[
                hvals - lvals,
                np.abs(hvals - prev_close),
                np.abs(lvals - prev_close),
            ],
            axis=1,
        )
        atr14 = pd.Series(tr).rolling(14).mean().iloc[-1]
        stats["atr_14"] = round(float(atr14), 4) if np.isfinite(atr14) else None
    else:
        stats["atr_14"] = None

    return stats


def print_summary(meta: Dict[str, Any], quick: Dict[str, Any]) -> None:
    def fmt_pct(x):
        return f"{x:.2f}%" if isinstance(x, (int, float)) and x is not None else "n/a"

    def fmt_num(x):
        return f"{x:,.4f}" if isinstance(x, float) else (f"{x:,}" if isinstance(x, int) else "n/a")

    print("")
    print("=== Snapshot ===")
    print(f"Symbol:            {meta.get('symbol','')}")
    print(f"Name:              {meta.get('shortName') or 'n/a'}")
    print(f"Exchange:          {meta.get('exchange') or 'n/a'}")
    print(f"Currency:          {meta.get('currency') or 'n/a'}")
    lp = meta.get("lastPrice")
    print(f"Last Price:        {lp if lp is not None else 'n/a'}")
    print(f"52w Range:         {meta.get('fiftyTwoWeekLow') or 'n/a'} - {meta.get('fiftyTwoWeekHigh') or 'n/a'}")
    print(f"Market Cap:        {humanize_int(meta.get('marketCap')) or 'n/a'}")
    print(f"PE (TTM/Fwd):      {(meta.get('trailingPE') or 'n/a')} / {(meta.get('forwardPE') or 'n/a')}")
    dy = meta.get("dividendYield")
    print(f"Dividend Yield:    {fmt_pct(dy*100) if isinstance(dy, (int, float)) else 'n/a'}")
    print(f"Beta:              {meta.get('beta') if meta.get('beta') is not None else 'n/a'}")
    print(f"Avg Volume:        {humanize_int(meta.get('avgVolume')) or 'n/a'}")
    print(f"Shares Out:        {humanize_int(meta.get('sharesOutstanding')) or 'n/a'}")
    print(f"Sector / Industry: {meta.get('sector') or 'n/a'} / {meta.get('industry') or 'n/a'}")
    print(f"Website:           {meta.get('website') or 'n/a'}")

    print("")
    print("=== Period Stats ===")
    print(f"Return:            {fmt_pct(quick.get('period_return_pct'))}")
    print(f"Volatility Ann:    {fmt_pct(quick.get('ann_vol_pct'))}")
    print(f"SMA 20 / 50 / 200: {fmt_num(quick.get('sma_20'))} / {fmt_num(quick.get('sma_50'))} / {fmt_num(quick.get('sma_200'))}")
    print(f"Avg Volume:        {humanize_int(quick.get('avg_volume')) or 'n/a'}")
    print(f"ATR(14):           {fmt_num(quick.get('atr_14'))}")
    print("")


def maybe_plot(df: pd.DataFrame, symbol: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Plotting skipped. Matplotlib import failed: {e}")
        return

    price = df["Adj Close"]
    vol = df["Volume"]

    plt.figure(figsize=(10, 5))
    plt.plot(price.index, price.values, label=f"{symbol} Adj Close")
    plt.title(f"{symbol} Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 3))
    plt.bar(vol.index, vol.values)
    plt.title(f"{symbol} Volume")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.tight_layout()
    plt.show()


def prompt_with_default(prompt: str, default: str) -> str:
    s = input(f"{prompt} [default {default}]: ").strip()
    return s if s else default


def yes_no(prompt: str, default_no: bool = True) -> bool:
    default = "n" if default_no else "y"
    s = input(f"{prompt} [y/n, default {default}]: ").strip().lower()
    if not s:
        return not default_no
    return s in {"y", "yes", "1", "true", "t"}


def main():
    print("Stock Lookup")
    query = input("Enter ticker: ").strip()
    if not query:
        print("No query provided. Exiting.")
        sys.exit(1)

    period = prompt_with_default("Period", "1y")
    interval = prompt_with_default("Interval", "1d")
    want_plot = yes_no("Show charts", default_no=False)

    try:
        symbol, longname, exchname = resolve_symbol(query)
    except ValueError as ve:
        print(str(ve))
        sys.exit(2)

    try:
        df = fetch_ohlcv(symbol, period=period, interval=interval)
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        sys.exit(3)

    meta = get_key_info(symbol)
    if longname and not meta.get("shortName"):
        meta["shortName"] = longname

    exch = meta.get("exchange") or exchname
    curr = meta.get("currency")
    if not _exchange_is_us_ca(exch, currency=curr):
        print(f"Resolved to non-US/CA listing: exchange='{exch}', currency='{curr}'. Only US/CA supported.")
        sys.exit(4)

    quick = compute_quick_stats(df)

    print_summary(meta, quick)
    print("=== OHLCV (tail) ===")
    try:
        print(df.tail().to_string())
    except Exception:
        print(df.tail())

    if want_plot:
        maybe_plot(df, symbol)

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)