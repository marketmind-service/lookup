import sys
import re
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import yfinance as yf
from state import LookupState

# optional name -> ticker search
try:
    from yahooquery import search as yq_search  # type: ignore

    HAVE_YQ = True
except Exception:
    HAVE_YQ = False

_INTERVAL_MINUTES = {
    "1m": 1,
    "2m": 2,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "90m": 90,
    "1h": 60,
    "4h": 240,
    "1d": 24 * 60,
    "5d": 5 * 24 * 60,
    "1wk": 7 * 24 * 60,
    "1mo": 30 * 24 * 60,
    "3mo": 90 * 24 * 60,
}

DEFAULT_PERIOD = "1mo"
DEFAULT_INTERVAL = "1d"


def _period_to_days(period: str) -> Optional[float]:
    m = re.fullmatch(r"(\d+)\s*([a-zA-Z]+)", period.strip().lower())
    if not m:
        return None

    value = int(m.group(1))
    unit = m.group(2)

    if unit in {"m", "min", "mins", "minute", "minutes"}:
        return value / (60 * 24)
    if unit in {"h", "hr", "hrs", "hour", "hours"}:
        return value / 24
    if unit in {"d", "day", "days"}:
        return float(value)
    if unit in {"wk", "wks", "w", "week", "weeks"}:
        return value * 7.0
    if unit in {"mo", "mon", "mons", "month", "months"}:
        return value * 30.0
    if unit in {"y", "yr", "yrs", "year", "years"}:
        return value * 365.0
    return None


def _infer_interval_from_period(period: str) -> str:
    days = _period_to_days(period)
    if days is None:
        return DEFAULT_INTERVAL
    if days <= 2:
        return "5m"
    if days <= 7:
        return "15m"
    if days <= 30:
        return "1h"
    if days <= 90:
        return "1d"
    if days <= 365:
        return "1d"

    return "1wk"


def _infer_period_from_interval(interval: str) -> str:
    mins = _INTERVAL_MINUTES.get(interval)
    if mins is None:
        return DEFAULT_PERIOD
    if mins <= 15:
        return "5d"
    if mins <= 60:
        return "1mo"
    if mins <= 240:
        return "3mo"
    if interval == "1d":
        return "1y"
    if interval in {"5d", "1wk"}:
        return "3y"
    if interval in {"1mo", "3mo"}:
        return "5y"

    return DEFAULT_PERIOD


def _resolve_period_interval(period: Optional[str], interval: Optional[str]) -> Tuple[str, str]:
    per = period
    ivl = interval

    # nothing in prompt
    if not per and not ivl:
        return DEFAULT_PERIOD, DEFAULT_INTERVAL

    # period only
    if per and not ivl:
        ivl = _infer_interval_from_period(per)
        return per, ivl

    # interval only
    if ivl and not per:
        per = _infer_period_from_interval(ivl)
        return per, ivl

    # both
    return per, ivl


# exchange code and name whitelists for US and CA
EXCHANGE_CODES_US = {"NMS", "NCM", "NGM", "NYQ", "ASE", "PCX", "BATS", "CBOE"}
EXCHANGE_NAMES_US = {"NASDAQ", "NYSE", "NYSE ARCA", "NYSE AMERICAN", "ARCA", "AMEX", "CBOE", "BATS"}

EXCHANGE_CODES_CA = {"TOR", "TSX", "TSXV", "V", "NEO", "CNQ", "CNSX"}
EXCHANGE_NAMES_CA = {"TORONTO", "TSX", "TSX VENTURE", "NEO", "CANADIAN SECURITIES", "CSE", "CBOE CANADA"}

CA_SUFFIXES = {"TO", "V", "NE", "CN"}

# ticker regex so words like nvidia do not look like tickers
TICKER_RE = re.compile(r"^[A-Z0-9]{1,5}(\.(TO|V|NE|CN))?$")


def is_likely_ticker(q: str) -> bool:
    raw = q.strip()
    if not raw or raw != raw.upper():
        return False

    return bool(TICKER_RE.match(raw.upper()))


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
        return 1 <= len(s) <= 5

    suf = s.rsplit(".", 1)[-1]
    return suf in CA_SUFFIXES


def resolve_symbol(query: str) -> Tuple[str, Optional[str], Optional[str]]:
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
        name_blob = f"{item.get('longname', '')} {item.get('shortname', '')}".lower()
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
    if isinstance(df_like, pd.Series):
        return pd.to_numeric(df_like, errors="coerce")

    if not isinstance(df_like, pd.DataFrame):
        raise ValueError(f"Unexpected type for {field_name}: {type(df_like)}")

    if df_like.shape[1] == 0:
        raise ValueError(f"No data in column '{field_name}'")

    if symbol in df_like.columns:
        s = df_like[symbol]
    else:
        best_col = df_like.count().idxmax()
        s = df_like[best_col]

    return pd.to_numeric(s, errors="coerce")


def fetch_ohlcv(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df_raw = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if df_raw is None or df_raw.empty:
        raise ValueError(f"Could not fetch OHLCV for {symbol} with period={period} interval={interval}")

    fields = {}
    for field in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if field in df_raw.columns:
            fields[field] = _pick_series(df_raw[field], symbol, field)
        else:
            if field == "Adj Close" and "Close" in df_raw.columns:
                fields[field] = _pick_series(df_raw["Close"], symbol, "Adj Close")
            else:
                raise ValueError(f"Field '{field}' missing in downloaded data for {symbol}")

    out = pd.DataFrame(fields).sort_index()
    out = out.dropna(how="all")
    if out.empty:
        raise ValueError(f"No valid rows for {symbol} after normalization")
    return out


def get_key_info(symbol: str) -> Dict[str, Any]:
    t = yf.Ticker(symbol)
    info: Dict[str, Any] = {}
    fast: Dict[str, Any] = {}

    try:
        info = t.get_info() or {}
    except Exception:
        pass

    try:
        fast = dict(t.fast_info) if hasattr(t, "fast_info") else {}
    except Exception:
        pass

    merged = {**fast, **info}
    return {
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
        "avgVolume": merged.get("averageVolume")
                     or merged.get("averageDailyVolume10Day")
                     or merged.get("threeMonthAverageVolume"),
        "sharesOutstanding": merged.get("sharesOutstanding"),
        "sector": merged.get("sector"),
        "industry": merged.get("industry"),
        "website": merged.get("website"),
        "lastPrice": merged.get("lastPrice")
                     or merged.get("regularMarketPrice")
                     or merged.get("currentPrice"),
    }


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

    # Annualized volatility
    if cvals.size > 2:
        rets = np.diff(cvals) / cvals[:-1]
        rets = rets[np.isfinite(rets)]
        stats["ann_vol_pct"] = (
            round(float(rets.std(ddof=1) * np.sqrt(252.0)) * 100, 2)
            if rets.size > 2 else None
        )
    else:
        stats["ann_vol_pct"] = None

    # SMAs
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
    print(f"Symbol:            {meta.get('symbol', '')}")
    print(f"Name:              {meta.get('shortName') or 'n/a'}")
    print(f"Exchange:          {meta.get('exchange') or 'n/a'}")
    print(f"Currency:          {meta.get('currency') or 'n/a'}")
    lp = meta.get("lastPrice")
    print(f"Last Price:        {lp if lp is not None else 'n/a'}")
    print(f"52w Range:         {meta.get('fiftyTwoWeekLow') or 'n/a'} - {meta.get('fiftyTwoWeekHigh') or 'n/a'}")
    print(f"Market Cap:        {humanize_int(meta.get('marketCap')) or 'n/a'}")
    print(f"PE (TTM/Fwd):      {(meta.get('trailingPE') or 'n/a')} / {(meta.get('forwardPE') or 'n/a')}")
    dy = meta.get("dividendYield")
    print(f"Dividend Yield:    {fmt_pct(dy * 100) if isinstance(dy, (int, float)) else 'n/a'}")
    print(f"Beta:              {meta.get('beta') if meta.get('beta') is not None else 'n/a'}")
    print(f"Avg Volume:        {humanize_int(meta.get('avgVolume')) or 'n/a'}")
    print(f"Shares Out:        {humanize_int(meta.get('sharesOutstanding')) or 'n/a'}")
    print(f"Sector / Industry: {meta.get('sector') or 'n/a'} / {meta.get('industry') or 'n/a'}")
    print(f"Website:           {meta.get('website') or 'n/a'}")

    print("")
    print("=== Period Stats ===")
    print(f"Return:            {fmt_pct(quick.get('period_return_pct'))}")
    print(f"Volatility Ann:    {fmt_pct(quick.get('ann_vol_pct'))}")
    print(
        f"SMA 20 / 50 / 200: {fmt_num(quick.get('sma_20'))} / "
        f"{fmt_num(quick.get('sma_50'))} / {fmt_num(quick.get('sma_200'))}"
    )
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


def search(state: LookupState) -> LookupState:
    print("Stock Lookup Agent")
    company = state.company
    if not company:
        print("No query provided in state. Exiting lookup node.")
        return state.model_copy(update={
            "error": "[search.py] No ticker provided. Exiting lookup node.",
        })

    period, interval = _resolve_period_interval(
        getattr(state, "period", None),
        getattr(state, "interval", None),
    )

    print(f"Using period={period}, interval={interval}")

    try:
        symbol, longname, exchname = resolve_symbol(company)
    except ValueError as ve:
        print(str(ve))
        return state.model_copy(update={
            "error": f"[search.py] {str(ve)}"
        })

    try:
        df = fetch_ohlcv(symbol, period=period, interval=interval)
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return state.model_copy(update={
            "error": f"[search.py] Error fetching data for {symbol}: {e}"
        })

    meta = get_key_info(symbol)
    if longname and not meta.get("shortName"):
        meta["shortName"] = longname

    exch = meta.get("exchange") or exchname
    curr = meta.get("currency")
    if not _exchange_is_us_ca(exch, currency=curr):
        print(f"Resolved to non-US/CA listing: exchange='{exch}', currency='{curr}'. Only US/CA supported.")
        return state.model_copy(update={
            "error": f"[search.py] Resolved to non-US/CA listing: exchange='{exch}', currency='{curr}'. Only US/CA supported."
        })

    quick = compute_quick_stats(df)

    print_summary(meta, quick)
    print("=== OHLCV (tail) ===")
    try:
        print(df.tail().to_string())
    except Exception:
        print(df.tail())

    want_plot = False
    if want_plot:
        maybe_plot(df, symbol)

    print("Done.")

    return state.model_copy(update={
        "company": meta.get("shortName"),
        "period": period,
        "interval": interval,

        # META (copied EXACTLY as returned)
        "symbol": meta.get("symbol"),
        "currency": meta.get("currency"),
        "exchange": meta.get("exchange"),
        "marketCap": meta.get("marketCap"),
        "trailingPE": meta.get("trailingPE"),
        "forwardPE": meta.get("forwardPE"),
        "dividendYield": meta.get("dividendYield"),
        "beta": meta.get("beta"),
        "fiftyTwoWeekLow": meta.get("fiftyTwoWeekLow"),
        "fiftyTwoWeekHigh": meta.get("fiftyTwoWeekHigh"),
        "avgVolume": meta.get("avgVolume"),
        "sharesOutstanding": meta.get("sharesOutstanding"),
        "sector": meta.get("sector"),
        "industry": meta.get("industry"),
        "website": meta.get("website"),
        "lastPrice": meta.get("lastPrice"),

        # QUICK STATS
        "period_return_pct": quick.get("period_return_pct"),
        "ann_vol_pct": quick.get("ann_vol_pct"),
        "sma_20": quick.get("sma_20"),
        "sma_50": quick.get("sma_50"),
        "sma_200": quick.get("sma_200"),
        "atr_14": quick.get("atr_14"),

        # TAIL OHLCV
        "tail_ohlcv": df.tail().to_dict(orient="index"),

        "error": None,
    })


def main():
    print("Stock Lookup")
    query = input("Enter ticker: ").strip()
    if not query:
        print("No query provided. Exiting.")
        sys.exit(1)

    period = prompt_with_default("Period", "1y")
    interval = prompt_with_default("Interval", "1d")
    want_plot = yes_no("Show charts", default_no=True)

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
