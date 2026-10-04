"""
Intraday scanner for NSE stocks, using the Upstox API.

Ranks stocks by *proxies* for unusual buying/selling activity. True "inflow"
(who bought, who sold) is not visible to retail traders, so this uses:

  rvol      volume so far today vs. average volume by the same time of day
            over previous sessions
  flow%     signed turnover proxy: each 5m bar's turnover counts as buying if
            the bar closed up, selling if down (crude tick rule), -100..+100
  vwap_dev  price vs. today's VWAP, in %
  orb       position vs. the opening range (first 15 minutes), in range widths
  chg       % change since previous close

The score is a weighted sum of z-scores. No LLM is involved in ranking.
LONG CANDIDATES are stocks passing every filter below; if none pass, the
scanner says so. Stops/targets/quantity are ATR-based risk arithmetic, not
predictions. This is a screener, not investment advice.

Setup (Upstox access tokens expire daily, generate a fresh one each morning):
  export UPSTOX_ACCESS_TOKEN="..."
  pip install -r requirements.txt

Usage:
  python scanner.py --demo                         # synthetic data, offline
  python scanner.py                                # live Upstox data
  python scanner.py --capital 100000 --risk-pct 0.5 --log runs.csv
  python scanner.py --watch 120                    # re-scan every 120 s
"""
import argparse
import datetime as dt
import gzip
import io
import json
import os
import sys
import time
import urllib.parse

import numpy as np
import pandas as pd

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    sys.exit("Install dependencies first: pip install -r requirements.txt")

console = Console(width=max(Console().width, 110))

UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL",
    "ITC", "LT", "AXISBANK", "KOTAKBANK", "HINDUNILVR", "BAJFINANCE", "MARUTI",
    "SUNPHARMA", "TATAMOTORS", "TATASTEEL", "NTPC", "ONGC", "ADANIENT",
    "WIPRO", "HCLTECH", "M&M", "POWERGRID", "COALINDIA",
]

WEIGHTS = {"rvol": 0.30, "flow_pct": 0.30, "vwap_dev": 0.15, "orb": 0.10, "chg": 0.15}
BAR_MIN = 5
HISTORY_DAYS = 10
MIN_PRICE = 50
MIN_AVG_TURNOVER = 5e7        # INR/day, skips illiquid names
ATR_BARS = 14

# Long-candidate filters (all must pass)
MIN_RVOL = 1.5
MIN_FLOW_PCT = 10.0
EARLIEST = dt.time(9, 30)     # opening range needs 15 min to form
LATEST = dt.time(14, 45)      # late entries leave no room for the trade

BASE = "https://api.upstox.com"
IST = "Asia/Kolkata"


# ------------------------------------------------------------ upstox ----
class Upstox:
    def __init__(self, token):
        import requests
        self.s = requests.Session()
        self.s.headers.update({"Accept": "application/json",
                               "Authorization": f"Bearer {token}"})

    def _get(self, url, retries=3):
        for i in range(retries):
            r = self.s.get(url, timeout=20)
            if r.status_code == 429:
                time.sleep(1.5 * (i + 1))
                continue
            if r.status_code == 401:
                sys.exit("Upstox returned 401: access token missing/expired. "
                         "Generate a fresh one (tokens expire daily).")
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"Rate limited: {url}")

    def instrument_keys(self, symbols):
        """Map trading symbol -> instrument_key from Upstox's public NSE master."""
        import requests
        raw = requests.get("https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
                           timeout=60).content
        items = json.loads(gzip.GzipFile(fileobj=io.BytesIO(raw)).read())
        want = set(symbols)
        return {i["trading_symbol"]: i["instrument_key"] for i in items
                if i.get("segment") == "NSE_EQ" and i.get("instrument_type") == "EQ"
                and i.get("trading_symbol") in want}

    def candles(self, key, today):
        k = urllib.parse.quote(key, safe="")
        frm = (today - dt.timedelta(days=HISTORY_DAYS * 2 + 4)).isoformat()
        to = (today - dt.timedelta(days=1)).isoformat()
        hist = self._get(f"{BASE}/v3/historical-candle/{k}/minutes/{BAR_MIN}/{to}/{frm}")
        live = self._get(f"{BASE}/v3/historical-candle/intraday/{k}/minutes/{BAR_MIN}")
        return parse_candles(hist, live)

    def fetch(self, symbols):
        keys = self.instrument_keys(symbols)
        missing = [s for s in symbols if s not in keys]
        if missing:
            console.print(f"[yellow]No instrument key for: {', '.join(missing)}[/]")
        today = dt.date.today()
        out = {}
        with console.status("Fetching Upstox candles...") as st:
            for n, (sym, key) in enumerate(keys.items(), 1):
                st.update(f"Fetching Upstox candles... {n}/{len(keys)} {sym}")
                try:
                    df = self.candles(key, today)
                    if len(df):
                        out[sym] = df
                except Exception as e:
                    console.print(f"[yellow]{sym}: {e}[/]")
                time.sleep(0.15)   # stay well under rate limits
        return out


def parse_candles(*payloads):
    """Upstox candle payloads -> DataFrame. Each candle is
    [timestamp, open, high, low, close, volume, oi], newest first."""
    rows = []
    for p in payloads:
        rows += (p.get("data") or {}).get("candles") or []
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame([r[:6] for r in rows],
                      columns=["ts", "open", "high", "low", "close", "volume"])
    ts = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(IST).dt.tz_localize(None)
    df = df.drop(columns="ts").set_index(ts).astype(float)
    return df[~df.index.duplicated(keep="last")].sort_index()


def synth_data(symbols, seed=11):
    """Offline synthetic 5m bars; last session is 'live' and cut off mid-day."""
    rng = np.random.default_rng(seed)
    out = {}
    days = pd.bdate_range(end=dt.date.today(), periods=HISTORY_DAYS + 1)
    for i, s in enumerate(symbols):
        price = rng.uniform(200, 3000)
        base_vol = rng.uniform(2e4, 2e5)
        frames = []
        for d_i, d in enumerate(days):
            idx = pd.date_range(d.replace(hour=9, minute=15),
                                d.replace(hour=15, minute=25), freq=f"{BAR_MIN}min")
            if d_i == len(days) - 1:
                idx = idx[:36]
            n = len(idx)
            curve = 1.6 - 1.2 * np.sin(np.pi * np.linspace(0, 1, n))
            vol = base_vol * curve * rng.lognormal(0, 0.25, n)
            drift = 0.0
            if d_i == len(days) - 1 and i % 4 == 0:
                vol *= 3.0
                drift = rng.choice([-1, 1]) * 0.0006
            close = price * np.cumprod(1 + rng.normal(drift, 0.0015, n))
            open_ = np.concatenate([[price], close[:-1]])
            high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.0004, n)))
            low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.0004, n)))
            frames.append(pd.DataFrame({"open": open_, "high": high, "low": low,
                                        "close": close, "volume": vol.astype(int)}, index=idx))
            price = close[-1]
        out[s] = pd.concat(frames)
    return out


# ----------------------------------------------------------- metrics ----
def compute_metrics(df):
    df = df.copy()
    df["date"] = df.index.date
    dates = sorted(df["date"].unique())
    if len(dates) < 3:
        return None
    today = dates[-1]
    t, prev = df[df["date"] == today], df[df["date"] < today]
    if len(t) < 3:
        return None

    last = t["close"].iloc[-1]
    prev_close = prev[prev["date"] == dates[-2]]["close"].iloc[-1]
    if last < MIN_PRICE:
        return None
    turn = (prev["close"] * prev["volume"]).groupby(prev["date"]).sum()
    if turn.tail(HISTORY_DAYS).mean() < MIN_AVG_TURNOVER:
        return None

    n = len(t)
    past = [prev[prev["date"] == d]["volume"].iloc[:n].sum()
            for d in dates[-(HISTORY_DAYS + 1):-1]]
    past = [p for p in past if p > 0]
    if not past:
        return None
    rvol = t["volume"].sum() / np.mean(past)

    typical = (t["high"] + t["low"] + t["close"]) / 3
    vwap = (typical * t["volume"]).sum() / max(t["volume"].sum(), 1)
    turnover = typical * t["volume"]
    flow_pct = (np.sign(t["close"] - t["open"]) * turnover).sum() / max(turnover.sum(), 1) * 100

    k = max(15 // BAR_MIN, 1)
    hi, lo = t["high"].iloc[:k].max(), t["low"].iloc[:k].min()
    width = max(hi - lo, 1e-9)
    orb = (last - hi) / width if last > hi else -(lo - last) / width if last < lo else 0.0

    both = pd.concat([prev.tail(ATR_BARS), t])
    pc = both["close"].shift()
    tr = pd.concat([both["high"] - both["low"], (both["high"] - pc).abs(),
                    (both["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.tail(ATR_BARS).mean()

    return {"price": last, "chg": (last / prev_close - 1) * 100, "rvol": rvol,
            "vwap": vwap, "vwap_dev": (last / vwap - 1) * 100, "flow_pct": flow_pct,
            "orb": float(np.clip(orb, -3, 3)), "atr": atr, "asof": t.index[-1]}


def zscore(s):
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0


def rank(metrics):
    df = pd.DataFrame(metrics).T
    for c in df.columns:
        if c != "asof":
            df[c] = df[c].astype(float)
    direction = np.sign(df["flow_pct"] + df["chg"])
    df["score"] = (WEIGHTS["rvol"] * zscore(df["rvol"]) * direction
                   + WEIGHTS["flow_pct"] * zscore(df["flow_pct"])
                   + WEIGHTS["vwap_dev"] * zscore(df["vwap_dev"])
                   + WEIGHTS["orb"] * zscore(df["orb"])
                   + WEIGHTS["chg"] * zscore(df["chg"]))
    return df.reindex(df["score"].abs().sort_values(ascending=False).index)


def long_candidates(df, capital, risk_pct):
    """Apply every long filter, then attach ATR-based levels and quantity."""
    c = df[(df["score"] > 0) & (df["rvol"] >= MIN_RVOL) & (df["flow_pct"] >= MIN_FLOW_PCT)
           & (df["vwap_dev"] > 0) & (df["orb"] > 0)].copy()
    c = c.sort_values("score", ascending=False)
    c["stop"] = c["price"] - 1.5 * c["atr"]
    c["target"] = c["price"] + 2.25 * c["atr"]          # 1 : 1.5 risk-reward
    risk_rupees = capital * risk_pct / 100
    per_share = (c["price"] - c["stop"]).clip(lower=0.05)
    c["qty"] = np.minimum(np.floor(risk_rupees / per_share), np.floor(capital / c["price"])).astype(int)
    return c


# ------------------------------------------------------------ render ----
def color(v, fmt="{:+.2f}"):
    return Text(fmt.format(v), style="green" if v > 0 else "red" if v < 0 else "dim")


def render(res, longs, args, demo):
    asof = res["asof"].iloc[0]
    now_t = asof.time()
    console.clear()
    tag = "  [bold yellow]DEMO DATA[/]" if demo else ""
    console.print(Panel.fit(
        f"[bold]INTRADAY SCANNER[/]{tag}\nas of [cyan]{asof:%d %b %Y %H:%M}[/]  |  "
        f"{len(res)} stocks scanned  |  capital ₹{args.capital:,.0f}, risk {args.risk_pct}%/trade",
        border_style="cyan"))

    if now_t < EARLIEST:
        console.print("[yellow]Too early: the opening range is still forming. Wait until 09:30.[/]")
    elif now_t > LATEST:
        console.print("[yellow]Late in the session: new entries have little time to work.[/]")

    t = Table(title="LONG CANDIDATES (all filters passed)", box=box.SIMPLE_HEAD,
              title_style="bold green", header_style="bold")
    for h in ["#", "Symbol", "Price", "Chg%", "RVol", "Flow%", "VWAP%", "Score", "Stop", "Target", "Qty"]:
        t.add_column(h, justify="left" if h == "Symbol" else "right", no_wrap=True)
    top = longs.head(args.top)
    for i, (sym, r) in enumerate(top.iterrows(), 1):
        t.add_row(str(i), f"[bold]{sym}[/]", f"{r.price:,.2f}", color(r.chg), f"{r.rvol:.1f}x",
                  color(r.flow_pct, "{:+.0f}"), color(r.vwap_dev), f"{r.score:.2f}",
                  f"[red]{r.stop:,.2f}[/]", f"[green]{r.target:,.2f}[/]", str(r.qty))
    if len(top):
        console.print(t)
    else:
        console.print("\n[bold green]LONG CANDIDATES[/]\n[dim]None. No stock passed every filter "
                      f"(RVol ≥ {MIN_RVOL}x, flow ≥ {MIN_FLOW_PCT:.0f}%, above VWAP, above opening range). "
                      "Sitting out is a valid call.[/]\n")

    s = Table(title="STRONGEST SELL-SIDE (avoid / watch for shorts)", box=box.SIMPLE_HEAD,
              title_style="bold red", header_style="bold")
    for h in ["Symbol", "Price", "Chg%", "RVol", "Flow%", "VWAP%", "Score"]:
        s.add_column(h, justify="left" if h == "Symbol" else "right", no_wrap=True)
    sells = res[(res["score"] < 0) & (res["rvol"] >= MIN_RVOL)].sort_values("score").head(5)
    for sym, r in sells.iterrows():
        s.add_row(f"[bold]{sym}[/]", f"{r.price:,.2f}", color(r.chg), f"{r.rvol:.1f}x",
                  color(r.flow_pct, "{:+.0f}"), color(r.vwap_dev), f"{r.score:.2f}")
    if len(sells):
        console.print(s)

    console.print("[dim]Flow% and RVol are proxies, not true order flow. Stop = price - 1.5 ATR, "
                  "target = price + 2.25 ATR, qty sized so a stop-out loses the risk %. "
                  "Screener output, not advice. Exit everything before the broker's square-off time.[/]")


def run_once(args, data_source):
    symbols = args.symbols or UNIVERSE
    data = data_source(symbols)
    metrics = {}
    for s, df in data.items():
        m = compute_metrics(df)
        if m:
            metrics[s] = m
    if not metrics:
        console.print("[red]No usable data (market closed, holiday, or fetch failed).[/]")
        return
    res = rank(metrics)
    longs = long_candidates(res, args.capital, args.risk_pct)
    render(res, longs, args, args.demo)
    if args.log:
        out = res.reset_index().rename(columns={"index": "symbol"})
        out["is_long_candidate"] = out["symbol"].isin(longs.index)
        out["logged_at"] = dt.datetime.now().isoformat(timespec="seconds")
        out.to_csv(args.log, mode="a", header=not os.path.exists(args.log), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="use synthetic data (no token needed)")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--capital", type=float, default=100000, help="INR available for trading")
    ap.add_argument("--risk-pct", type=float, default=0.5, help="max %% of capital lost per trade")
    ap.add_argument("--log", help="append every run to this CSV (for later backtesting)")
    ap.add_argument("--watch", type=int, metavar="SECS", help="re-scan every SECS seconds")
    ap.add_argument("--symbols", nargs="*", help="override the stock universe")
    args = ap.parse_args()

    if args.demo:
        source = synth_data
    else:
        token = os.environ.get("UPSTOX_ACCESS_TOKEN")
        if not token:
            sys.exit("Set UPSTOX_ACCESS_TOKEN first (or try --demo).")
        source = Upstox(token).fetch

    while True:
        run_once(args, source)
        if not args.watch:
            break
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
