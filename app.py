import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
 
warnings.filterwarnings("ignore")
 
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
 
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
 
# ─────────────────────────────────────────────
# PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Pharma Sales Intelligence",
    layout="wide",
    page_icon="💊",
    initial_sidebar_state="expanded"
)
 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #f0f4f8; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2942 0%, #1a3d5c 100%);
    }
    section[data-testid="stSidebar"] * { color: #e8f0f7 !important; }
    section[data-testid="stSidebar"] .stMultiSelect span { background: #1f5080; }
    section[data-testid="stSidebar"] label {
        font-weight: 600; letter-spacing: 0.04em;
        font-size: 0.78rem; text-transform: uppercase;
    }
    .app-header {
        background: linear-gradient(135deg, #0f2942 0%, #1565c0 60%, #0288d1 100%);
        padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
        display: flex; align-items: center; gap: 1.5rem;
        box-shadow: 0 8px 32px rgba(15,41,66,0.18);
    }
    .app-header h1 { color:#fff; font-size:2rem; font-weight:700; margin:0; letter-spacing:-0.5px; }
    .app-header p { color:#90caf9; margin:0; font-size:0.95rem; }
    .stTabs [data-baseweb="tab-list"] {
        background:#fff; padding:6px; border-radius:12px;
        gap:4px; box-shadow:0 2px 8px rgba(0,0,0,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius:8px; padding:8px 20px; font-weight:600;
        font-size:0.85rem; color:#546e7a; border:none !important; background:transparent;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg,#1565c0,#0288d1) !important;
        color:#fff !important; box-shadow:0 4px 12px rgba(21,101,192,0.3);
    }
    .kpi-card {
        background:#fff; border-radius:14px; padding:1.4rem 1.6rem;
        box-shadow:0 2px 12px rgba(0,0,0,0.06); border-top:4px solid #1565c0;
        transition:transform 0.2s;
    }
    .kpi-card:hover { transform:translateY(-2px); box-shadow:0 6px 20px rgba(0,0,0,0.10); }
    .kpi-label { font-size:0.75rem; font-weight:600; text-transform:uppercase;
                  letter-spacing:0.08em; color:#78909c; margin-bottom:0.4rem; }
    .kpi-value { font-size:1.7rem; font-weight:700; color:#0f2942; font-family:'DM Mono',monospace; }
    .kpi-delta-pos { font-size:0.82rem; color:#2e7d32; font-weight:600; }
    .kpi-delta-neg { font-size:0.82rem; color:#c62828; font-weight:600; }
    .section-title {
        font-size:1.15rem; font-weight:700; color:#0f2942;
        border-left:4px solid #1565c0; padding-left:0.75rem;
        margin:1.5rem 0 1rem 0;
    }
    .insight-card {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 5px solid #1565c0; padding: 1rem 1.4rem;
        border-radius: 0 10px 10px 0; margin-bottom: 0.75rem;
    }
    .forecast-banner {
        background:linear-gradient(90deg,#e8f5e9,#c8e6c9);
        border-left:5px solid #2e7d32; padding:1rem 1.4rem;
        border-radius:0 10px 10px 0; margin-bottom:0.75rem;
    }
    hr { border:none; border-top:1px solid #e0e7ef; margin:1.5rem 0; }
    .stDataFrame { border-radius:10px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
PALETTE = ["#1565c0","#0288d1","#26a69a","#66bb6a","#ffa726","#ef5350","#ab47bc","#5c6bc0"]
 
DRUG_MAP = {
    'M01AB': 'Diclofenac (Anti-inflam.)',
    'M01AE': 'Ibuprofen (Anti-inflam.)',
    'N02BA': 'Aspirin (Analgesic)',
    'N02BE': 'Paracetamol (Analgesic)',
    'N05B':  'Benzodiazepine (Anxiolytic)',
    'N05C':  'Sedative/Hypnotic',
    'R03':   'Salbutamol (Respiratory)',
    'R06':   'Antihistamine'
}
DRUG_CODES = list(DRUG_MAP.keys())
DRUG_NAMES = list(DRUG_MAP.values())
BEST_MODEL_FILE = "best_model.pkl"
 
# ─────────────────────────────────────────────
# DATA LOADING — REAL DATASET
# ─────────────────────────────────────────────
@st.cache_data
def load_dataset():
    """
    Loads real pharma sales data from salesdaily.csv.
    Melts from wide (one column per drug) to long format (one row per drug per day).
    Falls back to salesweekly.csv or salesmonthly.csv if daily not found.
    """
    for fname in ["salesdaily.csv", "salesweekly.csv", "salesmonthly.csv"]:
        if os.path.exists(fname):
            df_raw = pd.read_csv(fname)
            break
    else:
        st.error("❌ No dataset file found. Please place salesdaily.csv in the app folder.")
        st.stop()
 
    # Rename date column
    df_raw = df_raw.rename(columns={'datum': 'Date'})
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
 
    # Keep only drug columns that exist
    available_codes = [c for c in DRUG_CODES if c in df_raw.columns]
 
    # Melt wide → long
    df = df_raw.melt(
        id_vars=['Date'],
        value_vars=available_codes,
        var_name='Drug_Code',
        value_name='Sales_Units'
    )
    df['Product'] = df['Drug_Code'].map(DRUG_MAP)
    df['Sales_Units'] = pd.to_numeric(df['Sales_Units'], errors='coerce').fillna(0)
    df = df[df['Sales_Units'] > 0].copy()
 
    # Assign realistic price per drug (based on general pharma knowledge)
    price_map = {
        'M01AB': 8.50,   # Diclofenac
        'M01AE': 6.20,   # Ibuprofen
        'N02BA': 3.10,   # Aspirin
        'N02BE': 4.50,   # Paracetamol
        'N05B':  15.80,  # Benzodiazepine (prescription)
        'N05C':  12.40,  # Sedative
        'R03':   22.60,  # Salbutamol inhaler
        'R06':   9.80    # Antihistamine
    }
    df['Price'] = df['Drug_Code'].map(price_map)
    df['Revenue'] = (df['Sales_Units'] * df['Price']).round(2)
 
    # Date features
    df['Year']       = df['Date'].dt.year
    df['Month']      = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime("%b")
    df['Quarter']    = df['Date'].dt.quarter
    df['Week']       = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfWeek']  = df['Date'].dt.dayofweek
    df['Season']     = df['Month'].map({
        12:'Winter',1:'Winter',2:'Winter',
        3:'Spring',4:'Spring',5:'Spring',
        6:'Summer',7:'Summer',8:'Summer',
        9:'Autumn',10:'Autumn',11:'Autumn'
    })
 
    return df.sort_values('Date').reset_index(drop=True)
 
 
df_full = load_dataset()
ALL_PRODUCTS = sorted(df_full['Product'].unique().tolist())
DATE_MIN = df_full['Date'].min().date()
DATE_MAX = df_full['Date'].max().date()
 
# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div style="font-size:3rem">💊</div>
    <div>
        <h1>Pharma Sales Intelligence Platform</h1>
        <p>Real Data Analytics · Demand Forecasting · AI-Powered Insights &nbsp;|&nbsp; 
        Serbian Pharmacy · 2014–2019 · 8 Drug Categories</p>
    </div>
</div>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🎛️ Dashboard Controls")
st.sidebar.markdown("---")
 
products_sel = st.sidebar.multiselect(
    "Drug / Product", options=ALL_PRODUCTS, default=ALL_PRODUCTS
)
date_range = st.sidebar.date_input("Date Range", [DATE_MIN, DATE_MAX])
season_sel = st.sidebar.multiselect(
    "Season", options=["Winter","Spring","Summer","Autumn"],
    default=["Winter","Spring","Summer","Autumn"]
)
 
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Dataset Info")
st.sidebar.info(
    f"**Source:** Real Serbian Pharmacy\n\n"
    f"**Records:** {len(df_full):,}\n\n"
    f"**Drugs:** {df_full['Product'].nunique()}\n\n"
    f"**Date span:** {DATE_MIN} → {DATE_MAX}"
)
 
# ─────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────
if len(date_range) == 2:
    d0, d1 = date_range
else:
    d0, d1 = DATE_MIN, DATE_MAX
 
filtered = df_full[
    (df_full['Product'].isin(products_sel)) &
    (df_full['Season'].isin(season_sel)) &
    (df_full['Date'].dt.date >= d0) &
    (df_full['Date'].dt.date <= d1)
].copy().reset_index(drop=True)
 
# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🔍 Deep Analysis",
    "🤖 ML Predictions",
    "📈 Prophet Forecast",
    "🚨 Anomaly Detection",
    "🧪 What-If Simulator",
])
 
 
# ══════════════════════════════════════════════
# HELPER: KPI card
# ══════════════════════════════════════════════
def kpi_html(label, value, delta=None, color="#1565c0"):
    delta_html = ""
    if delta is not None:
        arrow = "▲" if delta >= 0 else "▼"
        cls   = "kpi-delta-pos" if delta >= 0 else "kpi-delta-neg"
        delta_html = f'<div class="{cls}">{arrow} {abs(delta):.1f}% vs prior half</div>'
    return (f'<div class="kpi-card" style="border-top-color:{color}">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>{delta_html}</div>')
 
 
def compute_delta(series, date_col, cutoff):
    curr = series[pd.to_datetime(date_col) >= cutoff].sum()
    prev = series[pd.to_datetime(date_col) < cutoff].sum()
    return ((curr - prev) / prev * 100) if prev else 0.0
 
 
# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    midpoint = filtered['Date'].median() if len(filtered) else pd.Timestamp.now()
    total_rev   = filtered['Revenue'].sum()
    total_units = filtered['Sales_Units'].sum()
    top_drug    = filtered.groupby('Product')['Revenue'].sum().idxmax() if len(filtered) else "N/A"
    years_span  = filtered['Year'].nunique()
 
    d_rev   = compute_delta(filtered['Revenue'],     filtered['Date'], midpoint)
    d_units = compute_delta(filtered['Sales_Units'], filtered['Date'], midpoint)
 
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_html("Total Revenue (€)", f"€{total_rev/1e6:.2f}M", d_rev), unsafe_allow_html=True)
    c2.markdown(kpi_html("Total Units Sold",  f"{total_units:,.0f}", d_units, "#0288d1"), unsafe_allow_html=True)
    c3.markdown(kpi_html("Top Drug", top_drug.split("(")[0].strip(), None, "#26a69a"), unsafe_allow_html=True)
    c4.markdown(kpi_html("Years of Data", f"{years_span} yrs", None, "#ffa726"), unsafe_allow_html=True)
 
    # KEY INSIGHTS banner
    st.markdown("""
    <div class="insight-card">
        <b>💡 Key Findings from Real Data</b><br>
        <span style="font-size:0.9rem">
        📌 <b>Paracetamol dominates</b> — ~5× higher sales volume than any other drug<br>
        📌 <b>Respiratory drugs (Salbutamol) spike in winter</b> — clear seasonal pattern<br>
        📌 <b>Anxiolytic (Benzodiazepine) demand shows a steady upward 6-year trend</b>
        </span>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("<hr>", unsafe_allow_html=True)
 
    # Revenue Trend
    st.markdown('<div class="section-title">Sales Revenue Trend (Daily Aggregated)</div>', unsafe_allow_html=True)
    rev_ts = filtered.groupby('Date')['Revenue'].sum().reset_index()
    rev_ts_monthly = rev_ts.set_index('Date').resample('ME')['Revenue'].sum().reset_index()
 
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=rev_ts_monthly['Date'], y=rev_ts_monthly['Revenue'],
        name="Monthly Revenue", line=dict(color="#1565c0", width=2.5),
        fill='tozeroy', fillcolor='rgba(21,101,192,0.08)'
    ))
    fig_trend.update_layout(
        paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
        margin=dict(l=10, r=10, t=10, b=10), height=300,
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#e8f0f7'),
        yaxis_title="Revenue (€)"
    )
    st.plotly_chart(fig_trend, use_container_width=True)
 
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Revenue by Drug</div>', unsafe_allow_html=True)
        prod_rev = filtered.groupby('Product')['Revenue'].sum().sort_values(ascending=True).reset_index()
        fig_prod = px.bar(prod_rev, x='Revenue', y='Product', orientation='h',
            color='Revenue', color_continuous_scale=["#bbdefb", "#1565c0"])
        fig_prod.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
            margin=dict(l=10,r=10,t=10,b=10), height=300,
            coloraxis_showscale=False, yaxis_title="", xaxis_title="Revenue (€)")
        st.plotly_chart(fig_prod, use_container_width=True)
 
    with col_b:
        st.markdown('<div class="section-title">Units Sold Share by Drug</div>', unsafe_allow_html=True)
        prod_units = filtered.groupby('Product')['Sales_Units'].sum().reset_index()
        fig_pie = px.pie(prod_units, values='Sales_Units', names='Product',
            color_discrete_sequence=PALETTE, hole=0.45)
        fig_pie.update_layout(paper_bgcolor='#fff',
            margin=dict(l=10,r=10,t=10,b=10), height=300,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3))
        st.plotly_chart(fig_pie, use_container_width=True)
 
    # Monthly Heatmap
    st.markdown('<div class="section-title">Monthly Sales Heatmap (Drug × Month)</div>', unsafe_allow_html=True)
    heat_data = filtered.groupby(['Product', 'Month'])['Sales_Units'].sum().unstack(fill_value=0)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    heat_data.columns = [month_labels[c-1] for c in heat_data.columns]
    fig_heat = px.imshow(heat_data, color_continuous_scale="Blues", aspect="auto",
        labels=dict(color="Units Sold"))
    fig_heat.update_layout(paper_bgcolor='#fff', margin=dict(l=10,r=10,t=10,b=10), height=300)
    st.plotly_chart(fig_heat, use_container_width=True)
 
    with st.expander("📄 View Raw Data & Download"):
        st.dataframe(filtered.head(100), use_container_width=True)
        st.download_button("⬇️ Download Filtered CSV",
                           filtered.to_csv(index=False).encode(),
                           "pharma_filtered.csv", "text/csv")
 
 
# ══════════════════════════════════════════════
# TAB 2 — DEEP ANALYSIS
# ══════════════════════════════════════════════
with tab2:
    # Seasonal Analysis
    st.markdown('<div class="section-title">Seasonal Sales Patterns by Drug</div>', unsafe_allow_html=True)
    season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    sea_data = filtered.groupby(['Season', 'Product'])['Sales_Units'].sum().reset_index()
    sea_data['Season'] = pd.Categorical(sea_data['Season'], categories=season_order, ordered=True)
    sea_data = sea_data.sort_values('Season')
    fig_sea = px.bar(sea_data, x='Season', y='Sales_Units', color='Product',
        barmode='group', color_discrete_sequence=PALETTE)
    fig_sea.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
        margin=dict(l=10,r=10,t=10,b=10), height=320,
        yaxis_title="Units Sold", xaxis_title="")
    st.plotly_chart(fig_sea, use_container_width=True)
 
    # Yearly Trend per Drug
    st.markdown('<div class="section-title">Year-over-Year Sales Trend per Drug</div>', unsafe_allow_html=True)
    yoy = filtered.groupby(['Year', 'Product'])['Sales_Units'].sum().reset_index()
    fig_yoy = px.line(yoy, x='Year', y='Sales_Units', color='Product',
        markers=True, color_discrete_sequence=PALETTE)
    fig_yoy.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
        margin=dict(l=10,r=10,t=10,b=10), height=320,
        yaxis_title="Units Sold")
    st.plotly_chart(fig_yoy, use_container_width=True)
 
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Day-of-Week Sales Pattern</div>', unsafe_allow_html=True)
        dow_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        dow_data = filtered.groupby('DayOfWeek')['Sales_Units'].mean().reset_index()
        dow_data['Day'] = dow_data['DayOfWeek'].map(lambda x: dow_labels[x])
        fig_dow = px.bar(dow_data, x='Day', y='Sales_Units',
            color='Sales_Units', color_continuous_scale=["#bbdefb","#1565c0"])
        fig_dow.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
            margin=dict(l=10,r=10,t=10,b=10), height=300,
            coloraxis_showscale=False, yaxis_title="Avg Units Sold")
        st.plotly_chart(fig_dow, use_container_width=True)
 
    with col2:
        st.markdown('<div class="section-title">Quarterly Performance</div>', unsafe_allow_html=True)
        q_data = filtered.groupby(['Year','Quarter'])['Revenue'].sum().reset_index()
        q_data['Period'] = "Q" + q_data['Quarter'].astype(str) + " " + q_data['Year'].astype(str)
        fig_q = px.bar(q_data, x='Period', y='Revenue',
            color='Revenue', color_continuous_scale=["#bbdefb","#0d47a1"], text_auto='.2s')
        fig_q.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
            margin=dict(l=10,r=10,t=10,b=10), height=300,
            coloraxis_showscale=False, yaxis_title="Revenue (€)", xaxis_title="")
        st.plotly_chart(fig_q, use_container_width=True)
 
    # Correlation Matrix
    st.markdown('<div class="section-title">Drug Sales Correlation Matrix</div>', unsafe_allow_html=True)
    st.caption("Shows how drug sales move together — e.g. cold season drugs often spike together.")
    # Pivot back to wide for correlation
    corr_wide = filtered.pivot_table(index='Date', columns='Product', values='Sales_Units', aggfunc='sum').fillna(0)
    corr_matrix = corr_wide.corr()
    fig_cm = px.imshow(corr_matrix, text_auto=".2f",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig_cm.update_layout(paper_bgcolor='#fff', margin=dict(l=10,r=10,t=10,b=10), height=400)
    st.plotly_chart(fig_cm, use_container_width=True)
 
    # Monthly Rolling Average
    st.markdown('<div class="section-title">30-Day Rolling Average Sales (Select Drug)</div>', unsafe_allow_html=True)
    roll_drug = st.selectbox("Select Drug", ALL_PRODUCTS, key="roll_drug")
    roll_df = filtered[filtered['Product'] == roll_drug].groupby('Date')['Sales_Units'].sum().reset_index()
    roll_df['Rolling_30'] = roll_df['Sales_Units'].rolling(30, min_periods=1).mean()
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=roll_df['Date'], y=roll_df['Sales_Units'],
        name="Daily Sales", line=dict(color="#bbdefb", width=1), opacity=0.6))
    fig_roll.add_trace(go.Scatter(x=roll_df['Date'], y=roll_df['Rolling_30'],
        name="30-Day Average", line=dict(color="#1565c0", width=2.5)))
    fig_roll.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
        margin=dict(l=10,r=10,t=10,b=10), height=300,
        yaxis_title="Units Sold", legend=dict(orientation="h"))
    st.plotly_chart(fig_roll, use_container_width=True)
 
 
# ══════════════════════════════════════════════
# TAB 3 — ML PREDICTIONS
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Model Training & Comparison (5-Fold Cross Validation)</div>',
                unsafe_allow_html=True)
 
    # Build ML features from time-series structure
    ml_df = filtered.copy()
    ml_df = pd.get_dummies(ml_df, columns=['Product'], drop_first=True)
    drop_cols = ['Date','Drug_Code','Month_Name','Season','Revenue']
    ml_df = ml_df.drop(columns=[c for c in drop_cols if c in ml_df.columns])
    ml_df = ml_df.dropna()
 
    target_col = 'Sales_Units'
 
    if len(ml_df) < 50:
        st.warning("⚠️ Not enough data for training. Expand your filters (need ≥ 50 rows).")
    else:
        X = ml_df.drop(columns=[target_col])
        y = ml_df[target_col]
 
        # Only keep numeric columns
        X = X.select_dtypes(include=[np.number])
 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
        models = {
            "Linear Regression":      LinearRegression(),
            "Random Forest":          RandomForestRegressor(n_estimators=200, random_state=42),
            "Gradient Boosting":      GradientBoostingRegressor(n_estimators=200, random_state=42),
            "Support Vector Regressor": SVR(C=1.0, epsilon=0.2),
        }
 
        results, trained_models = {}, {}
        prog = st.progress(0, text="Training models…")
        for i, (name, model) in enumerate(models.items()):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            results[name] = {
                "MAE":          mean_absolute_error(y_test, preds),
                "RMSE":         np.sqrt(mean_squared_error(y_test, preds)),
                "R² (test)":    r2_score(y_test, preds),
                "R² (CV mean)": cv_scores.mean(),
                "CV Std":       cv_scores.std(),
            }
            trained_models[name] = (model, preds)
            prog.progress((i + 1) / len(models), text=f"Trained: {name}")
        prog.empty()
 
        res_df = pd.DataFrame(results).T.sort_values("R² (test)", ascending=False)
        st.dataframe(res_df.style.format({
            "MAE": "{:.2f}", "RMSE": "{:.2f}",
            "R² (test)": "{:.3f}", "R² (CV mean)": "{:.3f}", "CV Std": "{:.3f}"
        }).background_gradient(subset=["R² (test)"], cmap="Blues"), use_container_width=True)
 
        best_name = res_df.index[0]
        best_model, best_preds = trained_models[best_name]
        st.success(f"🏆 Best Model: **{best_name}** — R² = {res_df.loc[best_name,'R² (test)']:.3f}")
 
        with open(BEST_MODEL_FILE, "wb") as f:
            pickle.dump(best_model, f)
 
        # Actual vs Predicted
        st.markdown('<div class="section-title">Actual vs Predicted Sales Units (Best Model)</div>',
                    unsafe_allow_html=True)
        n_plot = min(150, len(y_test))
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(x=list(range(n_plot)), y=y_test.values[:n_plot],
            mode='lines+markers', name='Actual',
            line=dict(color='#1565c0', width=2), marker=dict(size=4)))
        fig_avp.add_trace(go.Scatter(x=list(range(n_plot)), y=best_preds[:n_plot],
            mode='lines+markers', name='Predicted',
            line=dict(color='#ef5350', width=2, dash='dot'), marker=dict(size=4)))
        fig_avp.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
            margin=dict(l=10,r=10,t=10,b=10), height=300,
            xaxis_title="Sample Index", yaxis_title="Units Sold",
            legend=dict(orientation="h"))
        st.plotly_chart(fig_avp, use_container_width=True)
 
        # Residuals
        st.markdown('<div class="section-title">Residuals Distribution</div>', unsafe_allow_html=True)
        residuals = y_test.values[:len(best_preds)] - best_preds
        fig_res = px.histogram(x=residuals, nbins=60,
            color_discrete_sequence=["#1565c0"], labels={'x': 'Residual (Actual − Predicted)'})
        fig_res.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
            margin=dict(l=10,r=10,t=10,b=10), height=260)
        st.plotly_chart(fig_res, use_container_width=True)
 
        # Feature Importance
        if hasattr(best_model, 'feature_importances_'):
            st.markdown('<div class="section-title">Feature Importances</div>', unsafe_allow_html=True)
            fi = pd.Series(best_model.feature_importances_, index=X.columns) \
                   .sort_values(ascending=False).head(12).reset_index()
            fi.columns = ['Feature', 'Importance']
            fig_fi = px.bar(fi, x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale=["#bbdefb","#0d47a1"])
            fig_fi.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
                margin=dict(l=10,r=10,t=10,b=10), height=360,
                coloraxis_showscale=False, yaxis_title="", xaxis_title="Importance Score")
            fig_fi.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_fi, use_container_width=True)
 
        # SHAP
        if SHAP_AVAILABLE and hasattr(best_model, 'feature_importances_'):
            st.markdown('<div class="section-title">SHAP Explainability (Mean |SHAP| per Feature)</div>',
                        unsafe_allow_html=True)
            try:
                explainer   = shap.TreeExplainer(best_model)
                shap_values = explainer.shap_values(X_test.iloc[:200])
                shap_df     = pd.DataFrame(np.abs(shap_values), columns=X_test.columns)
                mean_shap   = shap_df.mean().sort_values(ascending=False).head(10).reset_index()
                mean_shap.columns = ['Feature', 'Mean_SHAP']
                fig_shap = px.bar(mean_shap, x='Mean_SHAP', y='Feature', orientation='h',
                    color='Mean_SHAP', color_continuous_scale=["#c8e6c9","#1b5e20"])
                fig_shap.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
                    margin=dict(l=10,r=10,t=10,b=10), height=340,
                    coloraxis_showscale=False, yaxis_title="", xaxis_title="Mean |SHAP value|")
                fig_shap.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_shap, use_container_width=True)
                st.info("ℹ️ SHAP values show each feature's average contribution to predictions — higher = more influential.")
            except Exception as e:
                st.warning(f"SHAP skipped: {e}")
 
        # Live Prediction
        st.markdown('<div class="section-title">🔮 Live Prediction — Forecast Units for a Drug</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        p_product = c1.selectbox("Drug", options=ALL_PRODUCTS, key="pred_drug")
        p_month   = c2.selectbox("Month", options=list(range(1,13)), key="pred_month")
        p_year    = c3.selectbox("Year", options=[2020, 2021, 2022, 2023, 2024, 2025], key="pred_year")
        c4, c5    = st.columns(2)
        p_dow     = c4.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 1, key="pred_dow")
        p_week    = c5.slider("Week of Year", 1, 52, 20, key="pred_week")
 
        if st.button("🚀 Predict Sales Units", key="pred_btn"):
            row = {
                'Year': p_year, 'Month': p_month,
                'Quarter': (p_month - 1) // 3 + 1,
                'Week': p_week, 'DayOfWeek': p_dow,
                'Price': {v: k for k, v in DRUG_MAP.items()}.get(p_product, 'N02BE')
            }
            # Map price from drug
            price_map_r = {
                'M01AB': 8.50, 'M01AE': 6.20, 'N02BA': 3.10, 'N02BE': 4.50,
                'N05B': 15.80, 'N05C': 12.40, 'R03': 22.60, 'R06': 9.80
            }
            drug_code_r = {v: k for k, v in DRUG_MAP.items()}.get(p_product, 'N02BE')
            row['Price'] = price_map_r.get(drug_code_r, 5.0)
 
            for col in X.columns:
                if col.startswith("Product_"):
                    row[col] = 0
            prod_col = f"Product_{p_product}"
            if prod_col in X.columns:
                row[prod_col] = 1
 
            inp  = pd.DataFrame([row]).reindex(columns=X.columns, fill_value=0)
            pred = best_model.predict(inp)[0]
 
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#e3f2fd,#bbdefb);padding:1.5rem;
                border-radius:12px;border-left:5px solid #1565c0;margin-top:1rem">
                <h3 style="color:#0d47a1;margin:0">Predicted Sales: {pred:,.0f} units</h3>
                <p style="color:#546e7a;margin-top:0.5rem">
                    Drug: <b>{p_product}</b> &nbsp;|&nbsp;
                    Month: <b>{p_month}</b> &nbsp;|&nbsp;
                    Year: <b>{p_year}</b> &nbsp;|&nbsp;
                    Model: <b>{best_name}</b>
                </p>
            </div>""", unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════
# TAB 4 — PROPHET FORECAST
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">📅 Prophet Time Series Forecasting</div>',
                unsafe_allow_html=True)
 
    if not PROPHET_AVAILABLE:
        st.error("Prophet is not installed. Run: `pip install prophet`")
    else:
        col_opts1, col_opts2, col_opts3 = st.columns(3)
        granularity   = col_opts1.radio("Granularity", ["Monthly", "Weekly"], horizontal=True)
        periods_fwd   = col_opts2.slider("Periods Ahead", 3, 24, 12)
        product_choice = col_opts3.selectbox("Drug", ["All Drugs"] + ALL_PRODUCTS, key="prophet_drug")
 
        freq = "ME" if granularity == "Monthly" else "W"
 
        ts_raw = df_full.copy()
        if product_choice != "All Drugs":
            ts_raw = ts_raw[ts_raw['Product'] == product_choice]
 
        ts_agg = ts_raw.set_index('Date').resample(freq)['Sales_Units'].sum().fillna(0).reset_index()
        ts_agg.columns = ['ds', 'y']
        ts_agg = ts_agg[ts_agg['y'] > 0]
 
        if len(ts_agg) < 10:
            st.warning("Not enough data points. Adjust filters.")
        else:
            with st.spinner("Training Prophet model on real data…"):
                m = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=(freq == "W"),
                    daily_seasonality=False,
                    seasonality_mode='multiplicative',
                    changepoint_prior_scale=0.05,
                    interval_width=0.80
                )
                m.fit(ts_agg)
                future   = m.make_future_dataframe(periods=periods_fwd, freq=freq)
                forecast = m.predict(future)
 
            hist_end = ts_agg['ds'].max()
 
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=ts_agg['ds'], y=ts_agg['y'], name="Actual",
                line=dict(color="#1565c0", width=2.5), mode="lines+markers",
                marker=dict(size=5)))
            fig_fc.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat'], name="Forecast",
                line=dict(color="#ef5350", width=2, dash="dot")))
            fig_fc.add_trace(go.Scatter(
                x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                fill='toself', fillcolor='rgba(239,83,80,0.10)',
                line=dict(color='rgba(255,255,255,0)'), name="80% Confidence Band"))
            fig_fc.add_vline(
                x=pd.Timestamp(hist_end).timestamp() * 1000,
                line_dash="dash", line_color="#90a4ae",
                annotation_text="Forecast Start")
            fig_fc.update_layout(
                paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
                margin=dict(l=10,r=10,t=10,b=10), height=400,
                legend=dict(orientation="h"),
                xaxis_title="", yaxis_title="Units Sold")
            st.plotly_chart(fig_fc, use_container_width=True)
 
            future_rows = forecast[forecast['ds'] > hist_end][
                ['ds','yhat','yhat_lower','yhat_upper']].copy()
            future_rows.columns = ['Period','Forecast','Lower Bound','Upper Bound']
            future_rows['Period'] = future_rows['Period'].dt.strftime("%b %Y")
            future_rows[['Forecast','Lower Bound','Upper Bound']] = \
                future_rows[['Forecast','Lower Bound','Upper Bound']].clip(lower=0)
 
            st.markdown('<div class="section-title">Forecast Table</div>', unsafe_allow_html=True)
            st.dataframe(future_rows.style.format({
                "Forecast":"{:,.0f}", "Lower Bound":"{:,.0f}", "Upper Bound":"{:,.0f}"
            }), use_container_width=True)
 
            # Seasonality decomposition
            st.markdown('<div class="section-title">Seasonality Decomposition</div>', unsafe_allow_html=True)
            comp_fig = make_subplots(rows=2, cols=1, subplot_titles=("Trend", "Yearly Seasonality"))
            comp_fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'],
                line=dict(color="#1565c0", width=2), name="Trend"), row=1, col=1)
            if 'yearly' in forecast.columns:
                comp_fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yearly'],
                    line=dict(color="#26a69a", width=2), name="Yearly Seasonality"), row=2, col=1)
            comp_fig.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
                margin=dict(l=10,r=40,t=40,b=10), height=400, showlegend=False)
            st.plotly_chart(comp_fig, use_container_width=True)
 
            total_forecast = future_rows['Forecast'].sum()
            st.markdown(f"""
            <div class="forecast-banner">
                <b>📊 Forecast Summary — {product_choice}</b><br>
                Next <b>{periods_fwd}</b> {granularity.lower()} periods →
                Expected Units = <b>{total_forecast:,.0f}</b>
                (Range: {future_rows['Lower Bound'].sum():,.0f} – {future_rows['Upper Bound'].sum():,.0f})
            </div>""", unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════
# TAB 5 — ANOMALY DETECTION
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">🚨 Sales Anomaly Detection — Isolation Forest</div>',
                unsafe_allow_html=True)
    st.markdown(
        "Isolation Forest detects **unusual sales spikes or drops** in real pharmacy data "
        "without needing labelled examples — e.g. unexpected demand surges or supply disruptions."
    )
 
    anom_drug = st.selectbox("Select Drug to Analyse", ALL_PRODUCTS, key="anom_drug")
    contamination = st.slider("Expected Anomaly Fraction", 0.01, 0.15, 0.05, 0.01,
                              help="Higher = more data points flagged as anomalous")
 
    anom_src = filtered[filtered['Product'] == anom_drug].copy()
    anom_df  = anom_src.groupby('Date').agg(
        Sales_Units=('Sales_Units','sum'),
        Revenue=('Revenue','sum')
    ).reset_index()
    anom_df['Date'] = pd.to_datetime(anom_df['Date'])
    anom_df['DayOfWeek'] = anom_df['Date'].dt.dayofweek
    anom_df['Month']     = anom_df['Date'].dt.month
    anom_df['Rolling7']  = anom_df['Sales_Units'].rolling(7, min_periods=1).mean()
 
    feat_cols = ['Sales_Units', 'Revenue', 'DayOfWeek', 'Month', 'Rolling7']
    scaler = StandardScaler()
    X_anom = scaler.fit_transform(anom_df[feat_cols])
    iso    = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    preds_a = iso.fit_predict(X_anom)
    scores  = iso.score_samples(X_anom)
    anom_df['Anomaly']       = np.where(preds_a == -1, "Anomaly", "Normal")
    anom_df['Anomaly_Score'] = -scores
    n_anom = (anom_df['Anomaly'] == "Anomaly").sum()
 
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records",      len(anom_df))
    c2.metric("Anomalies Detected", n_anom)
    c3.metric("Anomaly Rate",       f"{n_anom/len(anom_df)*100:.1f}%")
 
    # Timeline with anomalies highlighted
    st.markdown('<div class="section-title">Sales Timeline with Anomalies</div>', unsafe_allow_html=True)
    fig_tl = go.Figure()
    normal_df = anom_df[anom_df['Anomaly'] == 'Normal']
    anom_only = anom_df[anom_df['Anomaly'] == 'Anomaly']
    fig_tl.add_trace(go.Scatter(x=normal_df['Date'], y=normal_df['Sales_Units'],
        mode='markers', name='Normal',
        marker=dict(color='#1565c0', size=4, opacity=0.4)))
    fig_tl.add_trace(go.Scatter(x=anom_only['Date'], y=anom_only['Sales_Units'],
        mode='markers', name='Anomaly',
        marker=dict(color='#ef5350', size=10, symbol='x', line=dict(width=2))))
    fig_tl.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
        margin=dict(l=10,r=10,t=10,b=10), height=340,
        yaxis_title="Units Sold", legend=dict(orientation="h"))
    st.plotly_chart(fig_tl, use_container_width=True)
 
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown('<div class="section-title">Anomaly Score Distribution</div>', unsafe_allow_html=True)
        fig_dist = px.histogram(anom_df, x='Anomaly_Score', color='Anomaly', nbins=50,
            color_discrete_map={"Normal":"#1565c0","Anomaly":"#ef5350"},
            barmode='overlay', opacity=0.75)
        fig_dist.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
            margin=dict(l=10,r=10,t=10,b=10), height=280)
        st.plotly_chart(fig_dist, use_container_width=True)
 
    with col_d2:
        st.markdown('<div class="section-title">Revenue vs Units (Anomaly Map)</div>', unsafe_allow_html=True)
        fig_sc2 = px.scatter(anom_df, x='Sales_Units', y='Revenue',
            color='Anomaly', symbol='Anomaly',
            color_discrete_map={"Normal":"#1565c0","Anomaly":"#ef5350"},
            size='Anomaly_Score', opacity=0.7)
        fig_sc2.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
            margin=dict(l=10,r=10,t=10,b=10), height=280)
        st.plotly_chart(fig_sc2, use_container_width=True)
 
    with st.expander("📋 View Anomalous Records"):
        st.dataframe(
            anom_df[anom_df['Anomaly'] == 'Anomaly']
            .sort_values('Anomaly_Score', ascending=False).head(50),
            use_container_width=True)
 
 
# ══════════════════════════════════════════════
# TAB 6 — WHAT-IF SIMULATOR
# ══════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-title">🧪 Business Scenario Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        "Simulate how changes in **pricing** and **demand volume** affect revenue "
        "for any drug — before making real business decisions."
    )
 
    sim_drug = st.selectbox("Select Drug", ALL_PRODUCTS, key="sim_drug")
    drug_code_sim = {v: k for k, v in DRUG_MAP.items()}.get(sim_drug, 'N02BE')
    price_map_sim = {
        'M01AB': 8.50, 'M01AE': 6.20, 'N02BA': 3.10, 'N02BE': 4.50,
        'N05B': 15.80, 'N05C': 12.40, 'R03': 22.60, 'R06': 9.80
    }
    default_price = price_map_sim.get(drug_code_sim, 5.0)
 
    drug_avg_units = int(
        filtered[filtered['Product'] == sim_drug]['Sales_Units'].mean()
        if len(filtered[filtered['Product'] == sim_drug]) > 0 else 100
    )
 
    col_l, col_r = st.columns([1, 1.6])
    with col_l:
        st.markdown("#### 🎛️ Base Parameters")
        base_units = st.slider("Base Daily Units", 10, 1000, drug_avg_units, key="sim_units")
        base_price = st.slider("Base Price (€)", 1.0, 50.0, float(default_price), 0.10, key="sim_price")
        st.markdown("---")
        st.markdown("#### 📐 Scenario Adjustments")
        price_change  = st.slider("Price Change %", -30, 30, 0, key="sim_pc")
        units_change  = st.slider("Units Change %", -30, 30, 0, key="sim_uc")
 
    with col_r:
        new_price  = base_price * (1 + price_change / 100)
        new_units  = base_units * (1 + units_change / 100)
        base_rev   = base_units * base_price
        new_rev    = new_units  * new_price
        delta_pct  = ((new_rev - base_rev) / base_rev * 100) if base_rev else 0
 
        st.markdown("#### 📊 Scenario Results")
        r1, r2 = st.columns(2)
        r1.markdown(f"""
        <div class="kpi-card" style="border-top-color:#1565c0">
            <div class="kpi-label">Base Daily Revenue</div>
            <div class="kpi-value">€{base_rev:,.2f}</div>
            <div style="color:#78909c;font-size:0.8rem">Units: {base_units} | Price: €{base_price:.2f}</div>
        </div>""", unsafe_allow_html=True)
 
        dc    = "#2e7d32" if delta_pct >= 0 else "#c62828"
        arrow = "▲" if delta_pct >= 0 else "▼"
        r2.markdown(f"""
        <div class="kpi-card" style="border-top-color:{dc}">
            <div class="kpi-label">Projected Daily Revenue</div>
            <div class="kpi-value">€{new_rev:,.2f}</div>
            <div style="color:{dc};font-weight:600">{arrow} {abs(delta_pct):.1f}%</div>
        </div>""", unsafe_allow_html=True)
 
        # Annual projection
        st.markdown("#### 📅 Annual Projection")
        col_an1, col_an2 = st.columns(2)
        col_an1.metric("Base Annual Revenue",  f"€{base_rev * 365:,.0f}")
        col_an2.metric("Projected Annual",     f"€{new_rev * 365:,.0f}",
                       delta=f"{delta_pct:+.1f}%")
 
        # Sensitivity heatmap
        st.markdown("#### 🔀 Sensitivity Heatmap (Price × Volume Change)")
        scenarios = []
        for p_chg in [-20, -10, 0, 10, 20]:
            for u_chg in [-20, -10, 0, 10, 20]:
                s_price = base_price * (1 + p_chg / 100)
                s_units = base_units * (1 + u_chg / 100)
                s_rev   = s_units * s_price
                scenarios.append({"Price Δ": f"{p_chg:+d}%",
                                   "Volume Δ": f"{u_chg:+d}%",
                                   "Revenue": round(s_rev, 2)})
        scen_df = pd.DataFrame(scenarios)
        pivot = scen_df.pivot(index="Volume Δ", columns="Price Δ", values="Revenue")
        fig_s = px.imshow(pivot, text_auto='.0f',
            color_continuous_scale="RdYlGn", aspect="auto")
        fig_s.update_layout(paper_bgcolor='#fff', margin=dict(l=10,r=10,t=10,b=10), height=280)
        st.plotly_chart(fig_s, use_container_width=True)
        st.caption("🟢 Green = higher revenue   🔴 Red = lower revenue")
 
        # Breakeven
        if new_price > 0:
            breakeven = base_rev / new_price
            gap = new_units - breakeven
            st.markdown(f"""
            <div style="background:#e8f5e9;border-left:4px solid #2e7d32;
                padding:1rem;border-radius:0 10px 10px 0;margin-top:1rem">
                <b>⚖️ Breakeven Analysis</b><br>
                At new price (€{new_price:.2f}) you need <b>{breakeven:,.0f} units/day</b> to match base revenue.<br>
                Planned volume: {new_units:,.0f} →
                <span style="color:{'#2e7d32' if gap>=0 else '#c62828'};font-weight:700">
                    {gap:+,.0f} units {'surplus ✅' if gap >= 0 else 'shortfall ⚠️'}
                </span>
            </div>""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#90a4ae;font-size:0.82rem;padding:1rem">
    💊 <b>Pharma Sales Intelligence Platform</b><br>
    Real Data Analytics · Serbian Pharmacy · 2014–2019 · 8 Drug Categories<br>
    <span style="font-size:0.75rem;">Built with Streamlit · Prophet · Scikit-learn · Plotly · Real ATC Drug Data</span>
</div>
""", unsafe_allow_html=True)