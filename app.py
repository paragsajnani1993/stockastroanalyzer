import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ephem
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import defaultdict

# --- CONFIGURATION ---
st.set_page_config(page_title="Stock-Specific Astro Strategy", layout="wide")

# --- 0. STOCK UNIVERSE ---
MY_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS", 
    "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS", "BAJFINANCE.NS",
    "TATAMOTORS.NS", "SUNPHARMA.NS", "JSWSTEEL.NS", "HINDUNILVR.NS", "POWERGRID.NS", "^NSEI", "^NSEBANK"
]

# --- 1. ASTROLOGY ENGINE ---
planet_names = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 
                'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']
zodiacs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 
           'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']

def get_planet_state(check_date):
    obs = ephem.Observer()
    obs.lat, obs.lon = '19.0760', '72.8777' # Mumbai
    obs.date = check_date
    state = {}
    for name in planet_names:
        body = getattr(ephem, name)()
        body.compute(obs)
        lon = np.degrees(ephem.Ecliptic(body).lon)
        state[name] = {'lon': lon, 'sign_idx': int(lon / 30)}
    return state

def get_aspects(state_dict):
    found = set()
    names = list(state_dict.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p1, p2 = names[i], names[j]
            l1, l2 = state_dict[p1]['lon'], state_dict[p2]['lon']
            diff = abs(l1 - l2)
            if diff > 180: diff = 360 - diff
            if abs(diff - 0) < 3.0: found.add(f"{p1} conj {p2}")
            elif abs(diff - 180) < 3.0: found.add(f"{p1} opp {p2}")
            elif abs(diff - 120) < 3.0: found.add(f"{p1} tri {p2}")
            elif abs(diff - 90) < 3.0: found.add(f"{p1} sq {p2}")
    return found

@st.cache_data
def generate_event_universe(start_date, end_date):
    unique_events = set()
    curr_date = start_date
    prev_state = get_planet_state(curr_date - timedelta(days=1))
    
    while curr_date <= end_date:
        if hasattr(curr_date, 'to_pydatetime'): d_py = curr_date.to_pydatetime()
        else: d_py = curr_date
        curr_state = get_planet_state(d_py)
        
        for name in planet_names:
            if curr_state[name]['sign_idx'] != prev_state[name]['sign_idx']:
                unique_events.add(f"{name} enters {zodiacs[curr_state[name]['sign_idx'] % 12]}")
        
        asp_curr = get_aspects(curr_state)
        asp_prev = get_aspects(prev_state)
        unique_events.update(list(asp_curr - asp_prev))
        prev_state = curr_state
        curr_date += timedelta(days=1)
        
    return sorted(list(unique_events))

def get_dates_for_event(event_name, start_date, end_date):
    dates = []
    curr_date = start_date
    prev_state = get_planet_state(curr_date - timedelta(days=1))
    
    while curr_date <= end_date:
        if hasattr(curr_date, 'to_pydatetime'): d_py = curr_date.to_pydatetime()
        else: d_py = curr_date
        curr_state = get_planet_state(d_py)
        daily_events = []
        
        for name in planet_names:
            if curr_state[name]['sign_idx'] != prev_state[name]['sign_idx']:
                daily_events.append(f"{name} enters {zodiacs[curr_state[name]['sign_idx'] % 12]}")
        
        asp_curr = get_aspects(curr_state)
        asp_prev = get_aspects(prev_state)
        daily_events.extend(list(asp_curr - asp_prev))
        
        if event_name in daily_events:
            dates.append(curr_date)
        prev_state = curr_state
        curr_date += timedelta(days=1)
        
    return dates

# --- 2. ANALYSIS ENGINES ---
def analyze_portfolio_stats(ticker_list, event_dates, lookahead):
    leaderboard = []
    progress_bar = st.progress(0, text="Scanning Portfolio...")
    
    for idx, ticker in enumerate(ticker_list):
        try:
            df = yf.download(ticker, period="10y", progress=False) 
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if df.empty: continue
            
            highs, lows = [], []
            for event_date in event_dates:
                try:
                    loc_idx = df.index.get_indexer([event_date], method='nearest')[0]
                    if loc_idx == -1 or loc_idx >= len(df) - lookahead: continue
                    trigger_close = df['Close'].iloc[loc_idx]
                    window = df.iloc[loc_idx+1 : loc_idx+1+lookahead]
                    if window.empty: continue
                    
                    swing_h = ((window['High'].max() - trigger_close) / trigger_close) * 100
                    swing_l = ((window['Low'].min() - trigger_close) / trigger_close) * 100
                    highs.append(swing_h); lows.append(swing_l)
                except: pass
            
            if not highs: continue
            leaderboard.append({
                "Ticker": ticker,
                "Avg Swing High": np.mean(highs),
                "Avg Swing Low": np.mean(lows),
                "Bull Reliability": (sum(1 for h in highs if h > 2.0) / len(highs)) * 100,
                "Bear Reliability": (sum(1 for l in lows if l < -2.0) / len(lows)) * 100,
                "Occurrences": len(highs)
            })
        except: pass
        progress_bar.progress((idx + 1) / len(ticker_list))
    
    progress_bar.empty()
    return pd.DataFrame(leaderboard)

def get_stock_event_history(ticker, event_dates, lookahead):
    history = []
    try:
        df = yf.download(ticker, period="10y", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        for event_date in event_dates:
            try:
                loc_idx = df.index.get_indexer([event_date], method='nearest')[0]
                matched_date = df.index[loc_idx]
                if abs((matched_date - event_date).days) > 3: continue
                if loc_idx >= len(df) - lookahead: continue
                trigger_close = df['Close'].iloc[loc_idx]
                window = df.iloc[loc_idx+1 : loc_idx+1+lookahead]
                if window.empty: continue
                max_price = window['High'].max()
                min_price = window['Low'].min()
                swing_h = ((max_price - trigger_close) / trigger_close) * 100
                swing_l = ((min_price - trigger_close) / trigger_close) * 100
                history.append({
                    "Event Date": matched_date.date(),
                    "Trigger Close": trigger_close,
                    "Window High": max_price, "Window Low": min_price,
                    "Swing High %": swing_h, "Swing Low %": swing_l
                })
            except: pass
    except: return None, None
    if not history: return pd.DataFrame(), df
    return pd.DataFrame(history), df

# --- 3. UI ---
st.title("🧠 Stock-Specific Astro Strategy")
st.markdown("### Step 1: Choose Event. Step 2: Select Stock. Step 3: Get Strategy.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Backtest Settings")
    years_back = st.slider("Scan History (Years)", 1, 10, 5)
    lookahead = st.number_input("Swing Window (Days)", 1, 30, 7)
    end_d = datetime.now()
    start_d = end_d - timedelta(days=365 * years_back)
    st.divider()
    st.info(f"Scanning {len(MY_STOCKS)} Stocks from {start_d.date()}")

# --- MAIN ---
with st.spinner("Initializing Astro-Engine..."):
    event_options = generate_event_universe(start_d, end_d)

selected_event = st.selectbox("Select Astrological Event", event_options, index=0)

if st.button(f"Analyze '{selected_event}'"):
    st.session_state['current_event'] = selected_event
    dates = get_dates_for_event(selected_event, start_d, end_d)
    st.session_state['event_dates'] = dates
    st.write(f"Event Occurrences found: **{len(dates)}**")
    
    if len(dates) < 3: st.warning("⚠️ Low sample size.")
        
    lb_df = analyze_portfolio_stats(MY_STOCKS, dates, lookahead)
    st.session_state['leaderboard'] = lb_df

if 'leaderboard' in st.session_state and not st.session_state['leaderboard'].empty:
    lb_df = st.session_state['leaderboard']
    event_dates = st.session_state['event_dates']
    
    t1, t2, t3 = st.tabs(["🏆 Portfolio Overview", "🧠 Research & Strategy", "🔍 Visual Drill-Down"])
    
    # TAB 1: LEADERBOARD
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.success("### 🐂 Top Bullish Stocks")
            bulls = lb_df.sort_values("Avg Swing High", ascending=False).head(10)
            st.dataframe(bulls[['Ticker', 'Avg Swing High', 'Bull Reliability', 'Occurrences']].style.format({
                'Avg Swing High': '{:.2f}%', 'Bull Reliability': '{:.0f}%'
            }).background_gradient(cmap='Greens'), use_container_width=True)
        with c2:
            st.error("### 🐻 Top Bearish Stocks")
            bears = lb_df.sort_values("Avg Swing Low", ascending=True).head(10)
            st.dataframe(bears[['Ticker', 'Avg Swing Low', 'Bear Reliability', 'Occurrences']].style.format({
                'Avg Swing Low': '{:.2f}%', 'Bear Reliability': '{:.0f}%'
            }).background_gradient(cmap='Reds_r'), use_container_width=True)

    # TAB 2: DEEP RESEARCH & STRATEGY
    with t2:
        st.subheader(f"🔬 Strategic Report: {st.session_state['current_event']}")
        
        # --- STOCK SELECTOR FOR STRATEGY ---
        target_stock = st.selectbox("Select Stock for Deep Strategy", MY_STOCKS, key="strat_stock")
        
        # Retrieve data for this specific stock from leaderboard
        stock_data = lb_df[lb_df['Ticker'] == target_stock]
        
        if not stock_data.empty:
            row = stock_data.iloc[0]
            
            # 1. Metrics for THIS STOCK
            avg_upside = row['Avg Swing High']
            avg_downside = row['Avg Swing Low']
            bull_rel = row['Bull Reliability']
            bear_rel = row['Bear Reliability']
            
            # 2. Determine Bias
            bias = "NEUTRAL / CHOPPY"
            color = "gray"
            
            # Logic: Upside significantly greater than downside risk + High consistency
            if avg_upside > abs(avg_downside) * 1.3 and bull_rel > 60:
                bias = "BULLISH"
                color = "green"
            elif abs(avg_downside) > avg_upside * 1.3 and bear_rel > 60:
                bias = "BEARISH"
                color = "red"
            elif avg_upside > 4.0 and abs(avg_downside) > 4.0:
                bias = "HIGH VOLATILITY (STRADDLE)"
                color = "orange"
                
            # 3. Display Header
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"### Bias: :{color}[{bias}]")
            m2.metric("Avg Upside", f"+{avg_upside:.2f}%")
            m3.metric("Avg Downside", f"{avg_downside:.2f}%")
            m4.metric("Reliability", f"{max(bull_rel, bear_rel):.0f}%")
            
            st.divider()
            
            # 4. Generate Playbook
            col_strat, col_risk = st.columns(2)
            
            with col_strat:
                st.markdown(f"### 📜 Trading Playbook: {target_stock}")
                if bias == "BULLISH":
                    st.success(f"""
                    * **Primary Action:** Look for Long entries.
                    * **Entry Trigger:** Enter if {target_stock} breaks above the Trigger Day High.
                    * **Target:** {avg_upside:.1f}% to {avg_upside*1.2:.1f}% gain.
                    * **Why?** {target_stock} has rallied >2% in {bull_rel:.0f}% of historical instances.
                    """)
                elif bias == "BEARISH":
                    st.error(f"""
                    * **Primary Action:** Look for Short entries.
                    * **Entry Trigger:** Enter if {target_stock} breaks below the Trigger Day Low.
                    * **Target:** {avg_downside:.1f}% to {avg_downside*1.2:.1f}% drop.
                    * **Why?** {target_stock} has dropped >2% in {bear_rel:.0f}% of historical instances.
                    """)
                else:
                    st.info(f"""
                    * **Primary Action:** No clear directional edge.
                    * **Observation:** Upside (+{avg_upside:.1f}%) and Downside ({avg_downside:.1f}%) are roughly equal.
                    * **Strategy:** Wait for price action confirmation or choose a different stock from the Leaderboard.
                    """)

            with col_risk:
                st.markdown("### 🛡️ Risk Management")
                st.markdown(f"""
                * **Volatility Check:** Even if correct, this stock tends to swing **{avg_downside:.2f}% against you** (if Long) or **+{avg_upside:.2f}% against you** (if Short).
                * **Stop Loss Guide:** * **Long:** Set Stop Loss at least {abs(avg_downside):.1f}% below entry to avoid noise.
                    * **Short:** Set Stop Loss at least {avg_upside:.1f}% above entry.
                """)
        else:
            st.warning(f"No data available for {target_stock} on this event.")

    # TAB 3: DRILL DOWN (Unchanged)
    with t3:
        st.subheader(f"Visual History: {st.session_state.get('strat_stock', MY_STOCKS[0])}")
        # Use the stock selected in Tab 2 for continuity
        current_stock = st.session_state.get('strat_stock', MY_STOCKS[0])
        
        hist_df, price_df = get_stock_event_history(current_stock, event_dates, lookahead)
        if not hist_df.empty:
            st.markdown("### 📜 Historical Data Table")
            fmt_hist = {
                "Trigger Close": "{:.2f}", "Window High": "{:.2f}", "Window Low": "{:.2f}",
                "Swing High %": "{:.2f}%", "Swing Low %": "{:.2f}%"
            }
            st.dataframe(hist_df.style.format(fmt_hist).background_gradient(cmap='RdYlGn', subset=['Swing High %', 'Swing Low %']), use_container_width=True)
            
            st.markdown("### 📈 Price Chart with Triggers")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=price_df.index, open=price_df['Open'], high=price_df['High'], low=price_df['Low'], close=price_df['Close'], name='Price', opacity=0.3))
            fig.add_trace(go.Scatter(x=hist_df['Event Date'], y=hist_df['Trigger Close'], mode='markers', name=f"{st.session_state['current_event']}", marker=dict(color='gold', size=12, symbol='star', line=dict(width=1, color='black'))))
            fig.update_layout(height=500, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)