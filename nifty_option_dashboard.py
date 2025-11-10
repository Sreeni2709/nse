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
    for rec in data["records"]["data"]:
        if "CE" in rec:
            rec["CE"]["type"] = "Call"
            calls.append(rec["CE"])
        if "PE" in rec:
            rec["PE"]["type"] = "Put"
            puts.append(rec["PE"])
    return pd.DataFrame(calls), pd.DataFrame(puts)

# ---------------------------------------------
# Calculate Bias
# ---------------------------------------------
def calc_bias(calls_df, puts_df):
    call_oi_chg = calls_df["changeinOpenInterest"].sum()
    put_oi_chg = puts_df["changeinOpenInterest"].sum()
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
    cmax = calls_df.loc[calls_df["openInterest"].idxmax()]
    pmax = puts_df.loc[puts_df["openInterest"].idxmax()]
    return cmax, pmax

# ---------------------------------------------
# Main App
# ---------------------------------------------
def main():
    st.set_page_config("NSE Option Dashboard", layout="wide")
    st.sidebar.title("⚙️ Settings")

    # User Inputs
    symbol = st.sidebar.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"])
    refresh_rate = st.sidebar.slider("⏱️ Auto Refresh (sec)", 30, 180, 60)
    auto_refresh = st.sidebar.toggle("Enable Auto Refresh", True)

    # Initialize refresh history
    if "refresh_log" not in st.session_state:
        st.session_state.refresh_log = []

    # Fetch Data
    data = fetch_option_chain(symbol)
    if not data:
        st.error("⚠️ Unable to fetch NSE data.")
        return

    calls_df, puts_df = process_data(data)
    spot = data["records"]["underlyingValue"]

    expiry_list = sorted(calls_df["expiryDate"].unique())
    expiry = st.selectbox("Select Expiry", expiry_list)
    calls_df, puts_df = calls_df[calls_df.expiryDate == expiry], puts_df[puts_df.expiryDate == expiry]

    bias, diff = calc_bias(calls_df, puts_df)
    call_max, put_max = suggest_strike_prices(calls_df, puts_df)

    # Header Section
    st.markdown(f"### {symbol} : **{spot:.2f}** | Bias → 🟢 **{bias}**")
    st.caption(f"Updated : {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S IST')}")

    # Plot OI charts
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(calls_df, x="strikePrice", y="openInterest",
                     color_discrete_sequence=["#ef4444"], title="CALL Open Interest")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.bar(puts_df, x="strikePrice", y="openInterest",
                      color_discrete_sequence=["#22c55e"], title="PUT Open Interest")
        st.plotly_chart(fig2, use_container_width=True)

    # Combined OI Change
    st.subheader("📊 Combined OI Change (Call vs Put)")
    combined = pd.concat([calls_df.assign(side="Call"), puts_df.assign(side="Put")])
    st.plotly_chart(px.bar(combined, x="strikePrice", y="changeinOpenInterest", color="side",
                           barmode="group", title="Change in OI by Strike"), use_container_width=True)

    # Suggested Strikes
    st.subheader("🎯 Suggested Strikes (Max OI Levels)")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Max Call OI Strike", call_max["strikePrice"], f"OI {call_max['openInterest']}")
    with c2:
        st.metric("Max Put OI Strike", put_max["strikePrice"], f"OI {put_max['openInterest']}")

    # Bias Summary
    st.info(f"**{bias} Bias** | Net Put-Call OI Change = {diff:,}")

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
        st.error(f"⚠️ Unexpected Error: {e}")
        # Try rerun just in case
        try:
            st.rerun()
        except Exception:
            pass

