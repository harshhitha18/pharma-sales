# app.py — Pharma Sales Intelligence Platform
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
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
    .app-header p  { color:#90caf9; margin:0; font-size:0.95rem; }

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
    .kpi-label  { font-size:0.75rem; font-weight:600; text-transform:uppercase;
                  letter-spacing:0.08em; color:#78909c; margin-bottom:0.4rem; }
    .kpi-value  { font-size:1.7rem; font-weight:700; color:#0f2942; font-family:'DM Mono',monospace; }
    .kpi-delta-pos { font-size:0.82rem; color:#2e7d32; font-weight:600; }
    .kpi-delta-neg { font-size:0.82rem; color:#c62828; font-weight:600; }

    .section-title {
        font-size:1.15rem; font-weight:700; color:#0f2942;
        border-left:4px solid #1565c0; padding-left:0.75rem;
        margin:1.5rem 0 1rem 0;
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
DATA_CSV        = "pharma_sales_dataset.csv"
BEST_MODEL_FILE = "best_model.pkl"
PALETTE = ["#1565c0","#0288d1","#26a69a","#66bb6a","#ffa726","#ef5350","#ab47bc","#5c6bc0"]
PRODUCTS = ["PainRelief Tablet","Diabetes Control","Cough Syrup",
            "Vitamin C","Antibiotic Capsule","Antacid Tablet","Antihistamine"]
REGIONS  = ["North","South","East","West","Central"]

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def generate_synthetic_dataset(path=DATA_CSV, n=900, seed=42):
    np.random.seed(seed)
    from datetime import datetime as _dt
    start, end  = _dt(2023,1,1), _dt(2024,12,31)
    days_range  = (end - start).days
    dates = [start + pd.to_timedelta(int(np.random.randint(0, days_range)), unit='D') for _ in range(n)]
    df = pd.DataFrame({
        "Date":                 dates,
        "Product":              np.random.choice(PRODUCTS, size=n),
        "Region":               np.random.choice(REGIONS,  size=n),
        "Sales_Units":          np.random.randint(30, 800, size=n),
        "Price":                np.round(np.random.uniform(40, 600, size=n), 2),
        "Doctor_Prescriptions": np.random.randint(0, 50, size=n),
        "Discount_pct":         np.round(np.random.uniform(0, 20, size=n), 2),
    })
    df['Date']        = pd.to_datetime(df['Date']).dt.date
    df['Revenue']     = (df['Sales_Units'] * df['Price']).round(2)
    df['Net_Revenue'] = (df['Revenue'] * (1 - df['Discount_pct'] / 100)).round(2)
    df = df.sort_values('Date').reset_index(drop=True)
    df.to_csv(path, index=False)
    return df

@st.cache_data
def load_dataset(path=DATA_CSV):
    if not os.path.exists(path):
        df = generate_synthetic_dataset(path=path, n=900)
    else:
        df = pd.read_csv(path, parse_dates=["Date"])
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["Month"]      = pd.to_datetime(df["Date"]).dt.month
    df["Year"]       = pd.to_datetime(df["Date"]).dt.year
    df["Month_Name"] = pd.to_datetime(df["Date"]).dt.strftime("%b")
    df["Quarter"]    = pd.to_datetime(df["Date"]).dt.quarter
    return df

df_full = load_dataset()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div style="font-size:3rem">💊</div>
    <div>
        <h1>Pharma Sales Intelligence Platform</h1>
        <p>Advanced Analytics · Demand Forecasting · AI-Powered Insights</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🎛️ Dashboard Controls")
st.sidebar.markdown("---")
products_sel = st.sidebar.multiselect("Product(s)", options=PRODUCTS, default=PRODUCTS)
regions_sel  = st.sidebar.multiselect("Region(s)",  options=REGIONS,  default=REGIONS)
date_min = pd.to_datetime(df_full['Date']).min().date()
date_max = pd.to_datetime(df_full['Date']).max().date()
date_range = st.sidebar.date_input("Date Range", [date_min, date_max])
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Dataset Info")
st.sidebar.info(f"**Records:** {len(df_full):,}\n\n**Products:** {len(PRODUCTS)}\n\n**Regions:** {len(REGIONS)}\n\n**Date span:** {date_min} → {date_max}")

# ─────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────
filtered = df_full[
    (df_full['Product'].isin(products_sel)) &
    (df_full['Region'].isin(regions_sel)) &
    (pd.to_datetime(df_full['Date']).dt.date >= date_range[0]) &
    (pd.to_datetime(df_full['Date']).dt.date <= date_range[1])
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
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    def compute_delta(series, date_col, cutoff):
        curr = series[pd.to_datetime(date_col) >= cutoff].sum()
        prev = series[pd.to_datetime(date_col) <  cutoff].sum()
        return ((curr - prev) / prev * 100) if prev else 0.0

    midpoint   = pd.to_datetime(filtered['Date']).median() if len(filtered) else pd.Timestamp.now()
    total_rev  = filtered['Revenue'].sum()
    total_net  = filtered['Net_Revenue'].sum()
    total_units= filtered['Sales_Units'].sum()
    avg_disc   = filtered['Discount_pct'].mean()
    d_rev      = compute_delta(filtered['Revenue'],     filtered['Date'], midpoint)
    d_net      = compute_delta(filtered['Net_Revenue'], filtered['Date'], midpoint)
    d_units    = compute_delta(filtered['Sales_Units'], filtered['Date'], midpoint)

    def kpi_html(label, value, delta=None):
        delta_html = ""
        if delta is not None:
            arrow = "▲" if delta >= 0 else "▼"
            cls   = "kpi-delta-pos" if delta >= 0 else "kpi-delta-neg"
            delta_html = f'<div class="{cls}">{arrow} {abs(delta):.1f}% vs prior half</div>'
        return f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{delta_html}</div>'

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_html("Total Revenue",   f"₹{total_rev/1e5:.1f}L",  d_rev),   unsafe_allow_html=True)
    c2.markdown(kpi_html("Net Revenue",     f"₹{total_net/1e5:.1f}L",  d_net),   unsafe_allow_html=True)
    c3.markdown(kpi_html("Units Sold",      f"{total_units:,}",         d_units), unsafe_allow_html=True)
    c4.markdown(kpi_html("Avg Discount",    f"{avg_disc:.1f}%",         None),    unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Revenue trend
    st.markdown('<div class="section-title">Revenue Trend</div>', unsafe_allow_html=True)
    rev_ts = filtered.groupby('Date')[['Revenue','Net_Revenue']].sum().reset_index()
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=rev_ts['Date'], y=rev_ts['Revenue'],
        name="Gross Revenue", line=dict(color="#1565c0", width=2),
        fill='tozeroy', fillcolor='rgba(21,101,192,0.07)'))
    fig_trend.add_trace(go.Scatter(x=rev_ts['Date'], y=rev_ts['Net_Revenue'],
        name="Net Revenue", line=dict(color="#26a69a", width=2, dash='dot')))
    fig_trend.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
        margin=dict(l=10,r=10,t=10,b=10), height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#e8f0f7'))
    st.plotly_chart(fig_trend, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Revenue by Product</div>', unsafe_allow_html=True)
        prod_rev = filtered.groupby('Product')['Revenue'].sum().sort_values(ascending=True).reset_index()
        fig_prod = px.bar(prod_rev, x='Revenue', y='Product', orientation='h',
            color='Revenue', color_continuous_scale=["#bbdefb","#1565c0"])
        fig_prod.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
            margin=dict(l=10,r=10,t=10,b=10), height=280,
            coloraxis_showscale=False, yaxis_title="", xaxis_title="Revenue (₹)")
        st.plotly_chart(fig_prod, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Region Revenue Share</div>', unsafe_allow_html=True)
        reg_rev = filtered.groupby('Region')['Revenue'].sum().reset_index()
        fig_pie = px.pie(reg_rev, values='Revenue', names='Region',
            color_discrete_sequence=PALETTE, hole=0.45)
        fig_pie.update_layout(paper_bgcolor='#fff',
            margin=dict(l=10,r=10,t=10,b=10), height=280,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="section-title">Monthly Revenue Heatmap (Product × Month)</div>', unsafe_allow_html=True)
    heat_data = filtered.groupby(['Product','Month'])['Revenue'].sum().unstack(fill_value=0)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    heat_data.columns = [month_labels[c-1] for c in heat_data.columns]
    fig_heat = px.imshow(heat_data, color_continuous_scale="Blues", aspect="auto",
        labels=dict(color="Revenue (₹)"))
    fig_heat.update_layout(paper_bgcolor='#fff', margin=dict(l=10,r=10,t=10,b=10), height=280)
    st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander("📄 View Raw Data & Download"):
        st.dataframe(filtered.head(50), use_container_width=True)
        st.download_button("⬇️ Download CSV", filtered.to_csv(index=False).encode(),
                           "pharma_filtered.csv", "text/csv")

# ══════════════════════════════════════════════
# TAB 2 — DEEP ANALYSIS
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Quarterly Performance</div>', unsafe_allow_html=True)
    q_data = filtered.groupby(['Year','Quarter'])['Net_Revenue'].sum().reset_index()
    q_data['Period'] = "Q" + q_data['Quarter'].astype(str) + " " + q_data['Year'].astype(str)
    fig_q = px.bar(q_data, x='Period', y='Net_Revenue', color='Net_Revenue',
        color_continuous_scale=["#bbdefb","#0d47a1"], text_auto='.2s')
    fig_q.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
        margin=dict(l=10,r=10,t=10,b=10), height=280,
        coloraxis_showscale=False, yaxis_title="Net Revenue (₹)", xaxis_title="")
    st.plotly_chart(fig_q, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Price vs Sales Units</div>', unsafe_allow_html=True)
        fig_sc = px.scatter(filtered, x='Price', y='Sales_Units', color='Product',
            size='Net_Revenue', opacity=0.65, color_discrete_sequence=PALETTE,
            hover_data=['Region','Discount_pct'])
        fig_sc.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
            margin=dict(l=10,r=10,t=10,b=10), height=320)
        st.plotly_chart(fig_sc, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Discount Impact on Net Revenue</div>', unsafe_allow_html=True)
        disc_bins = pd.cut(filtered['Discount_pct'], bins=[0,5,10,15,20],
                           labels=["0-5%","5-10%","10-15%","15-20%"])
        disc_impact = filtered.groupby(disc_bins, observed=True)['Net_Revenue'].mean().reset_index()
        disc_impact.columns = ['Discount_Band','Avg_Net_Revenue']
        fig_disc = px.bar(disc_impact, x='Discount_Band', y='Avg_Net_Revenue',
            color='Avg_Net_Revenue', color_continuous_scale=["#1565c0","#ef5350"], text_auto='.0f')
        fig_disc.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
            margin=dict(l=10,r=10,t=10,b=10), height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_disc, use_container_width=True)

    st.markdown('<div class="section-title">Doctor Prescriptions vs Revenue (with Trendline)</div>', unsafe_allow_html=True)
    fig_corr = px.scatter(filtered, x='Doctor_Prescriptions', y='Net_Revenue',
        color='Product', trendline='ols', opacity=0.55, color_discrete_sequence=PALETTE)
    fig_corr.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
        margin=dict(l=10,r=10,t=10,b=10), height=320)
    st.plotly_chart(fig_corr, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-title">Correlation Matrix</div>', unsafe_allow_html=True)
        num_cols = ['Sales_Units','Price','Revenue','Net_Revenue','Doctor_Prescriptions','Discount_pct']
        corr_matrix = filtered[num_cols].corr()
        fig_cm = px.imshow(corr_matrix, text_auto=".2f",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig_cm.update_layout(paper_bgcolor='#fff', margin=dict(l=10,r=10,t=10,b=10), height=380)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col4:
        st.markdown('<div class="section-title">Region × Product Revenue Matrix</div>', unsafe_allow_html=True)
        rp_matrix = filtered.pivot_table(values='Revenue', index='Region',
                                          columns='Product', aggfunc='sum', fill_value=0)
        fig_rp = px.imshow(rp_matrix, color_continuous_scale="Blues", aspect="auto", text_auto='.0f')
        fig_rp.update_layout(paper_bgcolor='#fff', margin=dict(l=10,r=10,t=10,b=10), height=380)
        st.plotly_chart(fig_rp, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — ML PREDICTIONS
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Model Training & Comparison (with 5-Fold Cross Validation)</div>',
                unsafe_allow_html=True)

    ml_df = filtered.copy()
    ml_df = ml_df.drop(columns=['Date','Revenue','Year','Month_Name'], errors='ignore')
    ml_df = pd.get_dummies(ml_df, columns=['Product','Region'], drop_first=True)

    if 'Net_Revenue' not in ml_df.columns or len(ml_df) < 50:
        st.warning("⚠️ Not enough data for training. Expand your filters (need ≥ 50 rows).")
    else:
        X = ml_df.drop(columns=['Net_Revenue'])
        y = ml_df['Net_Revenue']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Linear Regression":        LinearRegression(),
            "Random Forest":            RandomForestRegressor(n_estimators=200, random_state=42),
            "Gradient Boosting":        GradientBoostingRegressor(n_estimators=200, random_state=42),
            "Support Vector Regressor": SVR(C=1.0, epsilon=0.2),
        }

        results, trained_models = {}, {}
        prog = st.progress(0, text="Training models…")
        for i, (name, model) in enumerate(models.items()):
            model.fit(X_train, y_train)
            preds    = model.predict(X_test)
            cv_scores= cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            results[name] = {
                "MAE":          mean_absolute_error(y_test, preds),
                "RMSE":         np.sqrt(mean_squared_error(y_test, preds)),
                "R² (test)":    r2_score(y_test, preds),
                "R² (CV mean)": cv_scores.mean(),
                "CV Std":       cv_scores.std(),
            }
            trained_models[name] = (model, preds)
            prog.progress((i+1)/len(models), text=f"Trained: {name}")
        prog.empty()

        res_df = pd.DataFrame(results).T.sort_values("R² (test)", ascending=False)
        st.dataframe(res_df.style.format({
            "MAE":"{:.2f}", "RMSE":"{:.2f}",
            "R² (test)":"{:.3f}", "R² (CV mean)":"{:.3f}", "CV Std":"{:.3f}"
        }).background_gradient(subset=["R² (test)"], cmap="Blues"), use_container_width=True)

        best_name  = res_df.index[0]
        best_model, best_preds = trained_models[best_name]
        st.success(f"🏆 Best Model: **{best_name}** — R² = {res_df.loc[best_name,'R² (test)']:.3f}")

        with open(BEST_MODEL_FILE, "wb") as f:
            pickle.dump(best_model, f)

        # Actual vs Predicted
        st.markdown('<div class="section-title">Actual vs Predicted (Best Model)</div>', unsafe_allow_html=True)
        n_plot = min(100, len(y_test))
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(x=list(range(n_plot)), y=y_test.values[:n_plot],
            mode='lines+markers', name='Actual',
            line=dict(color='#1565c0', width=2), marker=dict(size=5)))
        fig_avp.add_trace(go.Scatter(x=list(range(n_plot)), y=best_preds[:n_plot],
            mode='lines+markers', name='Predicted',
            line=dict(color='#ef5350', width=2, dash='dot'), marker=dict(size=5)))
        fig_avp.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
            margin=dict(l=10,r=10,t=10,b=10), height=300,
            xaxis_title="Sample Index", yaxis_title="Net Revenue (₹)",
            legend=dict(orientation="h"))
        st.plotly_chart(fig_avp, use_container_width=True)

        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            st.markdown('<div class="section-title">Feature Importances</div>', unsafe_allow_html=True)
            fi = pd.Series(best_model.feature_importances_, index=X.columns)\
                   .sort_values(ascending=False).head(12).reset_index()
            fi.columns = ['Feature','Importance']
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
                mean_shap.columns = ['Feature','Mean_SHAP']
                fig_shap = px.bar(mean_shap, x='Mean_SHAP', y='Feature', orientation='h',
                    color='Mean_SHAP', color_continuous_scale=["#c8e6c9","#1b5e20"])
                fig_shap.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
                    margin=dict(l=10,r=10,t=10,b=10), height=340,
                    coloraxis_showscale=False, yaxis_title="", xaxis_title="Mean |SHAP value|")
                fig_shap.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_shap, use_container_width=True)
                st.info("ℹ️ SHAP values show each feature's **average contribution** to predictions — higher = more influential.")
            except Exception as e:
                st.warning(f"SHAP skipped: {e}")
        elif not SHAP_AVAILABLE:
            st.info("📦 Install `shap` for explainability: `pip install shap`")

        # Live Prediction Form
        st.markdown('<div class="section-title">🔮 Live Prediction</div>', unsafe_allow_html=True)
        with st.form("pred_form"):
            c1, c2, c3 = st.columns(3)
            p_product  = c1.selectbox("Product", options=PRODUCTS)
            p_region   = c2.selectbox("Region",  options=REGIONS)
            p_month    = c3.selectbox("Month",   options=list(range(1,13)), index=datetime.now().month-1)
            c4, c5, c6 = st.columns(3)
            p_units    = c4.number_input("Sales Units",         min_value=1,   value=100)
            p_price    = c5.number_input("Price per Unit (₹)",  min_value=1.0, value=150.0)
            p_discount = c6.number_input("Discount %",          min_value=0.0, max_value=100.0, value=5.0)
            p_rx       = st.number_input("Doctor Prescriptions", min_value=0, value=10)
            submitted  = st.form_submit_button("🚀 Predict Net Revenue")

        if submitted:
            row = {'Sales_Units': p_units, 'Price': p_price, 'Month': p_month,
                   'Doctor_Prescriptions': p_rx, 'Discount_pct': p_discount,
                   'Quarter': (p_month-1)//3+1}
            for col in X.columns:
                if col.startswith("Product_") or col.startswith("Region_"):
                    row[col] = 0
            if f"Product_{p_product}" in X.columns: row[f"Product_{p_product}"] = 1
            if f"Region_{p_region}"   in X.columns: row[f"Region_{p_region}"]   = 1
            inp  = pd.DataFrame(row, index=[0]).reindex(columns=X.columns, fill_value=0)
            pred = best_model.predict(inp)[0]
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#e3f2fd,#bbdefb);padding:1.5rem;
                border-radius:12px;border-left:5px solid #1565c0;">
                <h3 style="color:#0d47a1;margin:0">Predicted Net Revenue: ₹{pred:,.2f}</h3>
                <p style="color:#546e7a;margin-top:0.5rem">
                    Gross = ₹{p_units*p_price:,.2f} &nbsp;|&nbsp;
                    Discount = {p_discount:.1f}% &nbsp;|&nbsp;
                    Model: <b>{best_name}</b>
                </p>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4 — PROPHET FORECAST
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">📅 Prophet Time Series Forecasting</div>', unsafe_allow_html=True)

    if not PROPHET_AVAILABLE:
        st.error("Prophet is not installed. Run: `pip install prophet`")
    else:
        col_opts1, col_opts2, col_opts3 = st.columns(3)
        granularity    = col_opts1.radio("Granularity",       ["Monthly","Weekly"], horizontal=True)
        periods_fwd    = col_opts2.slider("Periods Ahead",    3, 24, 6)
        product_choice = col_opts3.selectbox("Product Filter", ["All Products"] + PRODUCTS)

        freq = "M" if granularity == "Monthly" else "W"
        ts_raw = df_full.copy()
        ts_raw['Date'] = pd.to_datetime(ts_raw['Date'])
        if product_choice != "All Products":
            ts_raw = ts_raw[ts_raw['Product'] == product_choice]

        ts_agg = ts_raw.set_index('Date').resample(freq)['Net_Revenue'].sum().fillna(0).reset_index()
        ts_agg.columns = ['ds','y']
        ts_agg = ts_agg[ts_agg['y'] > 0]

        if len(ts_agg) < 10:
            st.warning("Not enough data points. Adjust filters.")
        else:
            with st.spinner("Training Prophet model…"):
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

            # Main chart
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
            fig_fc.add_vline(x=pd.Timestamp(hist_end).timestamp() * 1000, line_dash="dash",
                 line_color="#90a4ae", annotation_text="Forecast Start")
            fig_fc.update_layout(
                paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
                margin=dict(l=10,r=10,t=10,b=10), height=380,
                legend=dict(orientation="h"),
                xaxis_title="", yaxis_title="Net Revenue (₹)")
            st.plotly_chart(fig_fc, use_container_width=True)

            # Forecast table
            future_rows = forecast[forecast['ds'] > hist_end][
                ['ds','yhat','yhat_lower','yhat_upper']].copy()
            future_rows.columns = ['Period','Forecast','Lower Bound','Upper Bound']
            future_rows['Period'] = future_rows['Period'].dt.strftime("%b %Y")
            st.markdown('<div class="section-title">Forecast Table</div>', unsafe_allow_html=True)
            st.dataframe(future_rows.style.format({
                "Forecast":"{:,.0f}", "Lower Bound":"{:,.0f}", "Upper Bound":"{:,.0f}"}),
                use_container_width=True)

            # Decomposition
            st.markdown('<div class="section-title">Seasonality Decomposition</div>', unsafe_allow_html=True)
            comp_fig = make_subplots(rows=2, cols=1, subplot_titles=("Trend","Yearly Seasonality"))
            comp_fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'],
                line=dict(color="#1565c0", width=2), name="Trend"), row=1, col=1)
            if 'yearly' in forecast.columns:
                comp_fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yearly'],
                    line=dict(color="#26a69a", width=2), name="Yearly Seasonality"), row=2, col=1)
            comp_fig.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
                margin=dict(l=10,r=10,t=40,b=10), height=400, showlegend=False)
            st.plotly_chart(comp_fig, use_container_width=True)

            st.markdown(f"""
            <div class="forecast-banner">
            <b>📊 Forecast Summary ({product_choice})</b><br>
            Next <b>{periods_fwd}</b> {granularity.lower()} periods →
            Expected Net Revenue = <b>₹{future_rows['Forecast'].sum():,.0f}</b>
            (Range: ₹{future_rows['Lower Bound'].sum():,.0f} – ₹{future_rows['Upper Bound'].sum():,.0f})
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 5 — ANOMALY DETECTION
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">🚨 Sales Anomaly Detection — Isolation Forest</div>',
                unsafe_allow_html=True)
    st.markdown("Isolation Forest is an unsupervised ML algorithm that **isolates anomalies** "
                "rather than profiling normal data — ideal for detecting unusual sales events "
                "without labelled data.")

    contamination = st.slider("Expected Anomaly Fraction", 0.01, 0.15, 0.05, 0.01,
                              help="Higher = more data points flagged as anomalous")

    anom_df   = filtered[['Date','Sales_Units','Price','Revenue','Net_Revenue',
                           'Doctor_Prescriptions','Discount_pct']].copy()
    anom_df['Date'] = pd.to_datetime(anom_df['Date'])
    feat_cols = ['Sales_Units','Price','Net_Revenue','Doctor_Prescriptions','Discount_pct']

    scaler   = StandardScaler()
    X_anom   = scaler.fit_transform(anom_df[feat_cols])
    iso      = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    preds_a  = iso.fit_predict(X_anom)
    scores   = iso.score_samples(X_anom)

    anom_df['Anomaly']       = np.where(preds_a == -1, "Anomaly", "Normal")
    anom_df['Anomaly_Score'] = -scores

    n_anom = (anom_df['Anomaly'] == "Anomaly").sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records",      len(anom_df))
    col2.metric("Anomalies Detected", n_anom)
    col3.metric("Anomaly Rate",       f"{n_anom/len(anom_df)*100:.1f}%")

    # Scatter map
    st.markdown('<div class="section-title">Revenue vs Sales Units (Anomaly Map)</div>', unsafe_allow_html=True)
    fig_anom = px.scatter(anom_df, x='Sales_Units', y='Net_Revenue',
        color='Anomaly', symbol='Anomaly',
        color_discrete_map={"Normal":"#1565c0","Anomaly":"#ef5350"},
        size='Anomaly_Score', opacity=0.7,
        hover_data=['Date','Price','Discount_pct','Doctor_Prescriptions'])
    fig_anom.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
        margin=dict(l=10,r=10,t=10,b=10), height=380)
    st.plotly_chart(fig_anom, use_container_width=True)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        # Score distribution
        st.markdown('<div class="section-title">Score Distribution</div>', unsafe_allow_html=True)
        fig_dist = px.histogram(anom_df, x='Anomaly_Score', color='Anomaly', nbins=50,
            color_discrete_map={"Normal":"#1565c0","Anomaly":"#ef5350"},
            barmode='overlay', opacity=0.75)
        fig_dist.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
            margin=dict(l=10,r=10,t=10,b=10), height=300)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_d2:
        # Timeline
        st.markdown('<div class="section-title">Revenue Timeline</div>', unsafe_allow_html=True)
        fig_tl = go.Figure()
        normal_df = anom_df[anom_df['Anomaly'] == 'Normal']
        anom_only = anom_df[anom_df['Anomaly'] == 'Anomaly']
        fig_tl.add_trace(go.Scatter(x=normal_df['Date'], y=normal_df['Net_Revenue'],
            mode='markers', name='Normal',
            marker=dict(color='#1565c0', size=5, opacity=0.45)))
        fig_tl.add_trace(go.Scatter(x=anom_only['Date'], y=anom_only['Net_Revenue'],
            mode='markers', name='Anomaly',
            marker=dict(color='#ef5350', size=10, symbol='x', line=dict(width=2))))
        fig_tl.update_layout(paper_bgcolor='#fff', plot_bgcolor='#f8fbff',
            margin=dict(l=10,r=10,t=10,b=10), height=300,
            legend=dict(orientation="h"))
        st.plotly_chart(fig_tl, use_container_width=True)

    with st.expander("📋 View Anomalous Records"):
        st.dataframe(
            anom_df[anom_df['Anomaly']=='Anomaly']
            .sort_values('Anomaly_Score', ascending=False).head(50),
            use_container_width=True)

# ══════════════════════════════════════════════
# TAB 6 — WHAT-IF SIMULATOR
# ══════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-title">🧪 Business Scenario Simulator</div>', unsafe_allow_html=True)
    st.markdown("Adjust business levers to simulate how pricing and discount decisions "
                "impact **Net Revenue** — before committing to them.")

    col_l, col_r = st.columns([1, 1.6])
    with col_l:
        st.markdown("#### 🎛️ Base Parameters")
        base_units    = st.slider("Base Sales Units",   30, 800, 200)
        base_price    = st.slider("Base Price (₹)",     40, 600, 200)
        base_discount = st.slider("Base Discount %",    0.0, 20.0, 5.0, 0.5)
        st.markdown("---")
        st.markdown("#### 📐 Scenario Adjustments")
        price_change    = st.slider("Price Change %",       -30, 30, 0)
        units_change    = st.slider("Units Change %",       -30, 30, 0)
        discount_change = st.slider("Discount Change (pp)", -10.0, 10.0, 0.0, 0.5)

    with col_r:
        new_price    = base_price    * (1 + price_change/100)
        new_units    = base_units    * (1 + units_change/100)
        new_discount = min(max(base_discount + discount_change, 0), 100)

        base_revenue = base_units * base_price * (1 - base_discount/100)
        new_revenue  = new_units  * new_price  * (1 - new_discount/100)
        delta_pct    = ((new_revenue - base_revenue) / base_revenue * 100) if base_revenue else 0

        st.markdown("#### 📊 Scenario Results")
        r1, r2 = st.columns(2)
        r1.markdown(f"""
        <div class="kpi-card" style="border-top-color:#1565c0">
            <div class="kpi-label">Base Net Revenue</div>
            <div class="kpi-value">₹{base_revenue:,.0f}</div>
            <div style="color:#78909c;font-size:0.8rem">Units: {base_units} | Price: ₹{base_price} | Disc: {base_discount:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        dc = "#2e7d32" if delta_pct >= 0 else "#c62828"
        arrow = "▲" if delta_pct >= 0 else "▼"
        r2.markdown(f"""
        <div class="kpi-card" style="border-top-color:{dc}">
            <div class="kpi-label">Projected Net Revenue</div>
            <div class="kpi-value">₹{new_revenue:,.0f}</div>
            <div style="color:{dc};font-weight:600">{arrow} {abs(delta_pct):.1f}%</div>
        </div>""", unsafe_allow_html=True)

        # Multi-scenario heatmap
        st.markdown("#### 🔀 Sensitivity Heatmap (Price × Discount)")
        scenarios = []
        for p_chg in [-20,-10,0,10,20]:
            for d_chg in [-5,0,5]:
                s_price   = base_price    * (1 + p_chg/100)
                s_discount= min(max(base_discount + d_chg, 0), 100)
                s_revenue = base_units * s_price * (1 - s_discount/100)
                scenarios.append({"Price Δ": f"{p_chg:+d}%",
                                   "Discount Δ": f"{d_chg:+d}pp",
                                   "Net Revenue": round(s_revenue, 2)})
        scen_df = pd.DataFrame(scenarios)
        pivot   = scen_df.pivot(index="Discount Δ", columns="Price Δ", values="Net Revenue")
        fig_s   = px.imshow(pivot, text_auto='.0f',
            color_continuous_scale="RdYlGn", aspect="auto")
        fig_s.update_layout(paper_bgcolor='#fff', margin=dict(l=10,r=10,t=10,b=10), height=260)
        st.plotly_chart(fig_s, use_container_width=True)
        st.caption("🟢 Green = higher revenue &nbsp; 🔴 Red = lower revenue")

        # Breakeven
        if new_price > 0 and (1 - new_discount/100) > 0:
            breakeven = base_revenue / (new_price * (1 - new_discount/100))
            gap       = new_units - breakeven
            st.markdown(f"""
            <div style="background:#e8f5e9;border-left:4px solid #2e7d32;
                padding:1rem;border-radius:0 10px 10px 0;margin-top:1rem">
                <b>⚖️ Breakeven Analysis</b><br>
                At new price (₹{new_price:.0f}) you need <b>{breakeven:,.0f} units</b> to match base revenue.<br>
                Planned units: {new_units:,.0f} →
                <span style="color:{'#2e7d32' if gap>=0 else '#c62828'};font-weight:700">
                    {gap:+,.0f} units {'surplus ✅' if gap>=0 else 'shortfall ⚠️'}
                </span>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#90a4ae;font-size:0.82rem;padding:1rem">
    💊 <b>Pharma Sales Intelligence Platform</b> &nbsp;|&nbsp;
    Streamlit · Prophet · Scikit-learn · Plotly &nbsp;|&nbsp;
    Final Year Project
</div>
""", unsafe_allow_html=True)
