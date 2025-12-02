import re
import json
import textwrap
from typing import Optional, Tuple, List
from langchain_core.messages import SystemMessage, HumanMessage
from config import query, query2
from state import LookupState

SUPPORTED_INTERVALS = [
    "1m", "2m", "5m", "15m", "30m",
    "60m", "90m", "1h", "4h",
    "1d", "5d", "1wk", "1mo", "3mo",
]

# normalize common unit variants to canonical units
_UNIT_MAP = {
    # minutes
    "m": "m",
    "mn": "m",
    "mns": "m",
    "min": "m",
    "mins": "m",
    "minute": "m",
    "minutes": "m",

    # hours
    "h": "h",
    "hs": "h",
    "hr": "h",
    "hrs": "h",
    "hour": "h",
    "hours": "h",

    # days
    "d": "d",
    "ds": "d",
    "day": "d",
    "days": "d",

    # weeks
    "w": "wk",
    "wk": "wk",
    "wks": "wk",
    "week": "wk",
    "weeks": "wk",

    # months
    "mo": "mo",
    "mos": "mo",
    "mon": "mo",
    "mth": "mo",
    "mts": "mo",
    "mons": "mo",
    "mths": "mo",
    "mnth": "mo",
    "mnths": "mo",
    "month": "mo",
    "months": "mo",
}

_DURATION_RE = re.compile(
    r"(\d+)\s*("
    r"m|mn|mns|min|mins|minute|minutes|"
    r"h|hs|hr|hrs|hour|hours|"
    r"d|ds|day|days|"
    r"w|wk|wks|week|weeks|"
    r"mo|mos|mon|mth|mts|mons|mths|mnth|mnths|month|months"
    r")",
    re.IGNORECASE,
)


def _canonical_token(num: int, unit_raw: str) -> Optional[str]:
    unit = _UNIT_MAP.get(unit_raw.lower())
    if not unit:
        return None
    return f"{num}{unit}"


def _normalize_interval_token(token: str) -> Optional[str]:
    if not token:
        return None

    tok = token.strip().lower()
    if tok in SUPPORTED_INTERVALS:
        return tok

    m = re.fullmatch(r"(\d+)\s*([a-zA-Z]+)", tok)
    if not m:
        return None

    value = int(m.group(1))
    unit_raw = m.group(2).lower()
    unit = _UNIT_MAP.get(unit_raw)
    if not unit:
        return None

    day_candidates = {
        "1d": 1,
        "5d": 5,
        "1wk": 7,
        "1mo": 30,
        "3mo": 90,
    }

    minute_candidates = {
        "1m": 1,
        "2m": 2,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "60m": 60,
        "90m": 90,
        "1h": 60,
        "4h": 240,
    }

    if unit == "h":
        if value >= 6:
            days = value / 24.0
            best = min(day_candidates.keys(), key=lambda k: abs(day_candidates[k] - days))
            return best

        total_minutes = value * 60
        best = min(minute_candidates.keys(), key=lambda k: abs(minute_candidates[k] - total_minutes))
        return best

    if unit == "m":
        total_minutes = value
        best = min(minute_candidates.keys(), key=lambda k: abs(minute_candidates[k] - total_minutes))
        return best

    if unit == "d":
        days = value
    elif unit == "wk":
        days = value * 7
    elif unit == "mo":
        days = value * 30
    else:
        return None

    best = min(day_candidates.keys(), key=lambda k: abs(day_candidates[k] - days))
    return best


def _normalize_interval_for_yf(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    norm = _normalize_interval_token(token)
    return norm if norm in SUPPORTED_INTERVALS else None


def _extract_duration_tokens_from_prompt(prompt: str) -> List[str]:
    tokens: List[str] = []
    for match in _DURATION_RE.finditer(prompt):
        num_str, unit_str = match.groups()
        value = int(num_str)
        canon = _canonical_token(value, unit_str)
        if canon:
            tokens.append(canon)
    return tokens


def _canonical_from_raw(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    token = token.strip().lower()
    m = re.fullmatch(r"(\d+)\s*([a-zA-Z]+)", token)
    if not m:
        return token
    value = int(m.group(1))
    unit_raw = m.group(2)
    return _canonical_token(value, unit_raw) or token


async def extract_company(prompt: str) -> Optional[str]:
    messages = [
        SystemMessage(
            content=textwrap.dedent("""
            You extract ONE company or stock from a user's request.

            Output EXACTLY one JSON object with no spaces and no newline:
            {"company":"<VALUE>"} or {"company":null}

            Rules:
            - VALUE can be:
              * A stock ticker (NVDA, AAPL, RY.TO, SHOP.TO, QQQ)
              * OR an official company name (Nvidia, Apple, Royal Bank of Canada)
            - Fix obvious typos.
            - Map vague phrases to the most likely company, for example:
              "the iphone company" -> "Apple"
              "google stock" -> "Google" or "Alphabet"
              "NVDA stock" -> "NVDA"
            - If multiple companies appear, pick the MAIN one the user is asking about.
            - If you truly cannot infer any company, use null.
            - No explanations. No extra keys. No spaces anywhere.
            """).strip()
        ),
        HumanMessage(content=f"Prompt: {prompt}")
    ]

    response = query2.invoke(messages)
    raw = response.content if isinstance(response.content, str) else str(response.content)

    try:
        start, end = raw.find("{"), raw.rfind("}")
        obj = json.loads(raw[start:end + 1])

        comp = obj.get("company")
        if comp is None:
            return None

        comp = str(comp).strip()
        if not comp or comp.lower() == "null":
            return None

        plain = comp.replace(".", "").replace("-", "")
        if plain and plain.upper() == plain:
            return comp.upper()

        return comp

    except Exception:
        m = re.findall(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,3})?\b", prompt)
        if m:
            return m[0].strip().upper()
        stripped = prompt.strip()
        return stripped or None


async def time_args(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    message = [
        SystemMessage(content=textwrap.dedent("""
            You parse stock-chart requests. Output only:
            {"period":"<p>","interval":"<i>"}
            No spaces. No extra text
            
            period = chart range
            interval = candle width
            
            period units: d,wk,mo,y.
            interval units: m,h,d,wk,mo.
            
            Return EXACTLY one JSON object, with no spaces and no newline:
            {"period":"<p>","interval":"<i>"}
            
            Logic:
            - Extract all durations (like "30d", "24h", "5m", "3 weeks", "1 year").
            - Word conversions:
              minute(s)->1m, hour(s)->1h, day(s)->1d, week(s)->1wk, month(s)->1mo, year(s)->1y.
            - If ONE duration:
                If near words: candle(s), bar(s), tick(s), ticker, interval → interval.
                Else → period
            - If MULTIPLE durations:
                period = largest span
                interval = smallest span
            - If a duration is invalid for period (like minutes), convert to a reasonable range (e.g., "1d") or use null
            
            Examples:
            "nvda 30d chart" → {"period":"30d","interval":null}
            "nvda 30d, 24h candles" → {"period":"30d","interval":"24h"}
            "spy 1 year daily candles" → {"period":"1y","interval":"1d"}
            "tsla 1wk 5m candles" → {"period":"1wk","interval":"5m"}
        """).strip()),
        HumanMessage(content=f"Prompt: {prompt}")
    ]

    response = query.invoke(message)
    raw = response.content if isinstance(response.content, str) else str(response.content)
    print(f"raw time_args: {raw}")

    try:
        start, end = raw.find("{"), raw.rfind("}")
        obj = json.loads(raw[start:end + 1])

        per = obj.get("period")
        ivl = obj.get("interval")

        if isinstance(per, str) and per.lower() == "null":
            per = None
        if isinstance(ivl, str) and ivl.lower() == "null":
            ivl = None
    except Exception as e:
        print("time_args parse error:", e)
        return None, None

    prompt_tokens = _extract_duration_tokens_from_prompt(prompt)
    if len(prompt_tokens) == 1:
        single = prompt_tokens[0]
        canon_single = single
        canon_per = _canonical_from_raw(per)
        canon_ivl = _canonical_from_raw(ivl)

        # is the model already giving us a sensible interval?
        llm_interval_is_valid = canon_ivl in SUPPORTED_INTERVALS and canon_ivl != canon_per

        m_tok = re.fullmatch(r"(\d+)([a-zA-Z]+)", single)
        unit = None
        if m_tok:
            unit_raw = m_tok.group(2)
            unit = _UNIT_MAP.get(unit_raw.lower())

        interval_keywords = {
            "candle", "candles",
            "bar", "bars",
            "tick", "ticks",
            "ticker", "tickers",
            "interval",
        }
        context_interval_hint = False
        try:
            m_ctx = next(_DURATION_RE.finditer(prompt))
            tail = prompt[m_ctx.end():].lower()
            if any(kw in tail for kw in interval_keywords):
                context_interval_hint = True
        except StopIteration:
            pass

        if context_interval_hint:
            # only override with the single duration if the model did NOT already
            # infer a different valid interval (like 1wk for weekly candles).
            if not llm_interval_is_valid:
                ivl = single
                # only kill period if it is literally the same as the single token
                if canon_per == canon_single:
                    per = None
        else:
            if unit in {"m", "h"}:
                ivl = single
                if canon_per == single and (canon_ivl is None or canon_ivl == single):
                    per = None
            elif unit in {"d", "wk", "mo"}:
                per = single
                if canon_ivl == single:
                    ivl = None

    if isinstance(per, str):
        m = re.fullmatch(r"(\d+)\s*([a-zA-Z]+)", per.strip().lower())
        if m:
            unit_raw = m.group(2)
            unit = _UNIT_MAP.get(unit_raw.lower())
            if unit in {"m", "h"}:
                per = None

    return per, ivl


async def parse_input(state: LookupState) -> LookupState:
    print("parse_input")

    comp = await extract_company(state.prompt)

    per, ivl_raw = await time_args(state.prompt)
    ivl = _normalize_interval_for_yf(ivl_raw)

    print(f"Query (company or ticker): {comp}")
    print(f"Period: {per}")
    print(f"Interval (raw): {ivl_raw}")
    print(f"Interval (norm): {ivl}")

    updates: dict = {"company": comp}
    if per and per != "null":
        updates["period"] = per
    if ivl and ivl != "null":
        updates["interval"] = ivl

    return state.model_copy(update=updates)
