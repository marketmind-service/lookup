# app.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from stock_lookup import (
    resolve_symbol,
    fetch_ohlcv,
    get_key_info,
    compute_quick_stats,
    _exchange_is_us_ca,
)

app = FastAPI(title="Stock Lookup API")


@app.get("/api/stock")
def stock_lookup_api(
    query: str = Query(..., description="Ticker or company name"),
    period: str = Query("1y"),
    interval: str = Query("1d"),
):
    """
    Simple GET endpoint:
    /api/stock?query=NVDA&period=6mo&interval=1d
    """

    try:
        symbol, longname, exchname = resolve_symbol(query)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    try:
        df = fetch_ohlcv(symbol, period=period, interval=interval)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error fetching OHLCV: {e}")

    meta = get_key_info(symbol)
    if longname and not meta.get("shortName"):
        meta["shortName"] = longname

    exch = meta.get("exchange") or exchname
    curr = meta.get("currency")
    if not _exchange_is_us_ca(exch, currency=curr):
        raise HTTPException(
            status_code=400,
            detail=f"Resolved to non-US/CA listing: exchange='{exch}', currency='{curr}'. Only US/CA supported.",
        )

    quick = compute_quick_stats(df)

    # Small JSON response for the frontend
    return JSONResponse(
        {
            "meta": meta,
            "quick": quick,
            "tail_ohlcv": df.tail().to_dict(orient="index"),
        }
    )