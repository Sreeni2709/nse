# ---------------------------------------------
# NIFTY / BANKNIFTY Option Dashboard (Auto Refresh + Bias View)
# ---------------------------------------------
import time
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
import pytz

# added imports
import yfinance as yf
import numpy as np

# ---------------------------------------------
# Fetch Option Chain
# ---------------------------------------------
def fetch_option_chain(symbol):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.106 Safari/537.36"
    }
    s = requests.Session()
    # First request to set cookies
    r = s.get(url, headers=headers)
    s.cookies.update(r.cookies)
    response = s.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# ---------------------------------------------
# Process Data
# ---------------------------------------------
def process_data(data):
    calls, puts = [], []
    for rec in data.get("records", {}).get("data", []):
        if "CE" in rec:
            rec["CE"]["type"] = "Call"
            calls.append(rec["CE"])
        if "PE" in rec:
            rec["PE"]["type"] = "Put"
            puts.append(rec["PE"])
    calls_df = pd.DataFrame(calls) if calls else pd.DataFrame()
    puts_df = pd.DataFrame(puts) if puts else pd.DataFrame()
    return calls_df, puts_df

# ---------------------------------------------
# Calculate Bias
# ---------------------------------------------
def calc_bias(calls_df, puts_df):
    # handle empty dfs
    if calls_df.empty and puts_df.empty:
        return "Neutral", 0
    call_oi_chg = calls_df["changeinOpenInterest"].sum() if "changeinOpenInterest" in calls_df.columns else 0
    put_oi_chg = puts_df["changeinOpenInterest"].sum() if "changeinOpenInterest" in puts_df.columns else 0
    diff = put_oi_chg - call_oi_chg
    if diff > 0:
        return "Bullish", diff
    elif diff < 0:
        return "Bearish", diff
    else:
        return "Neutral", 0

# ---------------------------------------------
# Suggested Strikes
# ---------------------------------------------
def suggest_strike_prices(calls_df, puts_df):
    cmax = None
    pmax = None
    try:
        if not calls_df.empty and "openInterest" in calls_df.columns:
            cmax = calls_df.loc[calls_df["openInterest"].idxmax()]
    except Exception:
        cmax = None
    try:
        if not puts_df.empty and "openInterest" in puts_df.columns:
            pmax = puts_df.loc[puts_df["openInterest"].idxmax()]
    except Exception:
        pmax = None
    return cmax, pmax

# ---------------------------------------------
# RSI helper
# ---------------------------------------------
def compute_rsi(series: pd.Series, period: int = 7) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ---------------------------------------------
# VWAP helper
# ---------------------------------------------
def calc_vwap(df: pd.DataFrame) -> pd.Series:
    # df must have 'Close' and 'Volume'
    pv = (df['Close'] * df['Volume']).cumsum()
    v = df['Volume'].cumsum()
    return pv / v

# ---------------------------------------------
# Map symbol to yfinance ticker
# ---------------------------------------------
def symbol_to_yf_ticker(symbol: str):
    sym = symbol.upper()
    if sym == "NIFTY":
        return "^NSEI"
    if sym == "BANKNIFTY":
        return "^NSEBANK"
    if sym == "FINNIFTY":
        # FINNIFTY ticker may differ; default to NIFTY for safety
        return "^NSEI"
    # fallback
    return "^NSEI"

# ---------------------------------------------
# Main App
# ---------------------------------------------
def main():
    st.set_page_config("NSE Option Dashboard", layout="wide")
    st.sidebar.title("⚙️ Settings")

    # User Inputs
    symbol = st.sidebar.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"])
    refresh_rate = st.sidebar.slider("⏱️ Auto Refresh (sec)", 30, 180, 60)
    auto_refresh = st.sidebar.checkbox("Enable Auto Refresh", True)

    # Initialize refresh history
    if "refresh_log" not in st.session_state:
        st.session_state.refresh_log = []

    # Fetch Data
    data = fetch_option_chain(symbol)
    if not data:
        st.error("⚠️ Unable to fetch NSE data.")
        return

    calls_df, puts_df = process_data(data)
    spot = data.get("records", {}).get("underlyingValue", None)

    # Expiry selection guard
    expiry_list = []
    if not calls_df.empty and "expiryDate" in calls_df.columns:
        expiry_list = sorted(calls_df["expiryDate"].unique())
    expiry = st.selectbox("Select Expiry", expiry_list) if expiry_list else None
    if expiry is not None:
        calls_df = calls_df[calls_df.expiryDate == expiry]
        puts_df = puts_df[puts_df.expiryDate == expiry]

    bias, diff = calc_bias(calls_df, puts_df)
    call_max, put_max = suggest_strike_prices(calls_df, puts_df)

    # Header Section
    spot_text = f"{spot:.2f}" if spot is not None else "—"
    st.markdown(f"### {symbol} : **{spot_text}** | Bias → 🟢 **{bias}**")
    st.caption(f"Updated : {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S IST')}")

    # Plot OI charts
    col1, col2 = st.columns(2)
    with col1:
        if not calls_df.empty and "strikePrice" in calls_df.columns and "openInterest" in calls_df.columns:
            fig = px.bar(calls_df, x="strikePrice", y="openInterest",
                         color_discrete_sequence=["#ef4444"], title="CALL Open Interest")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Calls data not available for plotting.")
    with col2:
        if not puts_df.empty and "strikePrice" in puts_df.columns and "openInterest" in puts_df.columns:
            fig2 = px.bar(puts_df, x="strikePrice", y="openInterest",
                          color_discrete_sequence=["#22c55e"], title="PUT Open Interest")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Puts data not available for plotting.")

    # Combined OI Change
    st.subheader("📊 Combined OI Change (Call vs Put)")
    if (not calls_df.empty and "strikePrice" in calls_df.columns) or (not puts_df.empty and "strikePrice" in puts_df.columns):
        combined = pd.concat([calls_df.assign(side="Call"), puts_df.assign(side="Put")])
        if "changeinOpenInterest" in combined.columns:
            st.plotly_chart(px.bar(combined, x="strikePrice", y="changeinOpenInterest", color="side",
                                   barmode="group", title="Change in OI by Strike"), use_container_width=True)
        else:
            st.info("changeinOpenInterest column not available to plot combined OI change.")
    else:
        st.info("Insufficient data to plot combined OI change.")

    # Suggested Strikes
    st.subheader("🎯 Suggested Strikes (Max OI Levels)")
    c1, c2 = st.columns(2)
    with c1:
        if call_max is not None:
            st.metric("Max Call OI Strike", call_max.get("strikePrice", "—"), f"OI {call_max.get('openInterest', '—')}")
        else:
            st.metric("Max Call OI Strike", "—", "No data")
    with c2:
        if put_max is not None:
            st.metric("Max Put OI Strike", put_max.get("strikePrice", "—"), f"OI {put_max.get('openInterest', '—')}")
        else:
            st.metric("Max Put OI Strike", "—", "No data")

    # Bias Summary
    st.info(f"**{bias} Bias** | Net Put-Call OI Change = {diff:,}")

    # ---------------------------------------------
    # 🧠 Technical Strategy (EMA + RSI + VWAP)
    # ---------------------------------------------
    st.subheader("📈 Technical Strategy Confirmation (EMA + RSI + VWAP)")

    import json
    from requests.exceptions import RequestException
    # Check if current time is within market hours
    from datetime import time as dtime

    now_ist = datetime.now(pytz.timezone("Asia/Kolkata")).time()
    if now_ist < dtime(9, 15) or now_ist > dtime(15, 30):
        st.warning("📴 Market is closed. Live intraday data will be available only between 09:15–15:30 IST.")
    else:
        hist = fetch_nse_intraday(nse_symbol)

    def fetch_nse_intraday(symbol_name: str):
        """Fetch intraday data from NSE official chart API (robust version)"""
        session = requests.Session()
        base_url = "https://www.nseindia.com"
        chart_url = f"{base_url}/api/chart-databyindex?index={symbol_name.replace(' ', '%20')}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": base_url,
            "Connection": "keep-alive"
        }

        try:
            # Initialize session with NSE
            session.get(base_url, headers=headers, timeout=10)
            response = session.get(chart_url, headers=headers, timeout=15)

            if response.status_code != 200:
                raise ValueError(f"HTTP {response.status_code} from NSE")

            data = response.json()
            if "grapthData" not in data:
                raise ValueError("Invalid JSON response")

            df = pd.DataFrame(data["grapthData"], columns=["Timestamp", "Close"])
            df["Datetime"] = pd.to_datetime(df["Timestamp"], unit="ms", errors="coerce")
            df["Close"] = df["Close"].astype(float)
            df["Volume"] = 1.0  # dummy volume for VWAP
            df = df.dropna().set_index("Datetime")
            return df

        except (RequestException, json.JSONDecodeError, ValueError) as e:
            st.warning(f"⚠️ NSE Intraday fetch failed: {e}")
            return pd.DataFrame()

    try:
        # Map NSE index name
        if symbol == "NIFTY":
            nse_symbol = "NIFTY 50"
        elif symbol == "BANKNIFTY":
            nse_symbol = "NIFTY BANK"
        elif symbol == "FINNIFTY":
            nse_symbol = "NIFTY FIN SERVICE"
        else:
            nse_symbol = "NIFTY MIDCAP SELECT"

        hist = fetch_nse_intraday(nse_symbol)
        if hist.empty:
            st.warning("⚠️ Could not fetch intraday data from NSE.")
        else:
            # Compute Indicators
            hist["EMA8"] = hist["Close"].ewm(span=8, adjust=False).mean()
            hist["EMA21"] = hist["Close"].ewm(span=21, adjust=False).mean()
            hist["RSI7"] = compute_rsi(hist["Close"], period=7)
            hist["VWAP"] = (hist["Close"] * hist["Volume"]).cumsum() / hist["Volume"].cumsum()

            latest = hist.iloc[-1]
            bullish = (latest["Close"] > latest["VWAP"]) and (latest["EMA8"] > latest["EMA21"]) and (latest["RSI7"] > 55)
            bearish = (latest["Close"] < latest["VWAP"]) and (latest["EMA8"] < latest["EMA21"]) and (latest["RSI7"] < 45)

            if bullish:
                tech_signal = "BUY CE"
            elif bearish:
                tech_signal = "BUY PE"
            else:
                tech_signal = "WAIT"

            st.metric("📊 Technical Signal", tech_signal)
            st.write(f"""
            **Latest Price:** {latest['Close']:.2f}  
            **EMA(8):** {latest['EMA8']:.2f}  **EMA(21):** {latest['EMA21']:.2f}  
            **RSI(7):** {latest['RSI7']:.1f}  **VWAP:** {latest['VWAP']:.2f}
            """)

            if tech_signal == "BUY CE" and bias == "Bullish":
                combined_signal = "CONFIRMED BUY CE"
            elif tech_signal == "BUY PE" and bias == "Bearish":
                combined_signal = "CONFIRMED BUY PE"
            elif tech_signal == "WAIT":
                combined_signal = "WAIT"
            else:
                combined_signal = "MISMATCH / WAIT"

            st.success(f"🧩 Combined View: {combined_signal}")

            fig_price = px.line(hist.tail(60).reset_index(), x="Datetime", y=["Close", "EMA8", "EMA21", "VWAP"],
                                title=f"{nse_symbol} Live Intraday Chart (NSE)", template="plotly_dark")
            st.plotly_chart(fig_price, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Could not fetch intraday data: {e}")




    # Refresh Log
    now_time = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S')
    st.session_state.refresh_log.append(now_time)
    st.caption(f"🕒 Last {len(st.session_state.refresh_log)} refreshes: {', '.join(st.session_state.refresh_log[-5:])}")

    # Auto Refresh
    if auto_refresh:
        st.caption(f"🔄 Auto-refreshing data every {refresh_rate} seconds...")
        time.sleep(refresh_rate)
        try:
            st.rerun()  # Streamlit >= 1.39
        except AttributeError:
            st.experimental_rerun()  # Backward compatibility

# ---------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # st may not be available in some contexts; guard print too
        try:
            st.error(f"⚠️ Unexpected Error: {e}")
        except Exception:
            print("Unexpected Error:", e)
        # Try a safe rerun if possible
        try:
            st.rerun()
        except Exception:
            pass
