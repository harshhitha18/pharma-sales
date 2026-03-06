# 💊 Pharma Sales Intelligence Platform

> Advanced Analytics · Demand Forecasting · AI-Powered Insights

A powerful, interactive business intelligence dashboard built for pharmaceutical sales analytics. It combines machine learning, time series forecasting, and real-time visual analytics to deliver actionable insights across products and regions.

---

## 🚀 Features

- **📊 Overview Dashboard** — KPIs, revenue trends, regional breakdowns, and product performance at a glance
- **🔍 Deep Analysis** — Correlation analysis, doctor prescription trends, and OLS trendlines
- **🤖 ML Predictions** — Trained machine learning model for sales prediction using Scikit-learn
- **📈 Prophet Forecasting** — Time series demand forecasting with Facebook Prophet
- **🎛️ Interactive Filters** — Filter by product, region, and custom date range

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Forecasting | Prophet |
| Visualization | Plotly Express, Matplotlib |
| Statistical Analysis | Statsmodels |

---

## 📁 Project Structure

```
pharma-sales/
│
├── app.py                   # Main Streamlit application
├── best_model.pkl           # Pre-trained ML model
├── pharma_sales_dataset.csv # Sales dataset (900 records)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/harshitha18/pharma-sales.git
cd pharma-sales
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the application**
```bash
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
plotly
scikit-learn
prophet
matplotlib
statsmodels
```

Or install all at once:
```bash
pip install streamlit pandas numpy plotly scikit-learn prophet matplotlib statsmodels
```

---

## 📊 Dataset Overview

| Property | Value |
|---|---|
| Total Records | 900 |
| Products | 7 |
| Regions | Multiple |
| Date Range | 2023 – 2024 |

---

## 📸 Screenshots

> Dashboard with product filters, revenue trends, ML predictions, and Prophet forecasting.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
