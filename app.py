import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="AI Financial Advisor", layout="wide")
st.title("💰 AI-Based Financial Planning Advisor")

# -----------------------------
# Global UI polish
# -----------------------------
st.markdown("""
<style>

/* Bigger section headers */
h1, h2, h3 {
    font-weight: 600;
}

/* Remove Plotly mode bar */
.js-plotly-plot .plotly .modebar {
    display: none;
}

/* Metric cards styling */
[data-testid="metric-container"] {
    background-color: #0e1117;
    border-radius: 12px;
    padding: 12px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load models safely
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

kmeans_model = joblib.load(os.path.join(BASE_DIR, "models", "kmeans_recommender.pkl"))
recommender_scaler = joblib.load(os.path.join(BASE_DIR, "models", "recommender_scaler.pkl"))
growth_model = joblib.load(os.path.join(BASE_DIR, "models", "growth_prediction_model.pkl"))
growth_scaler = joblib.load(os.path.join(BASE_DIR, "models", "growth_scaler.pkl"))

# -----------------------------
# Load stock behavior dataset
# -----------------------------
market_df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "market_cleaned.csv")
)
index_df = market_df[["Stock Index", "Close Price"]].dropna()

cluster_features = [
    "Age",
    "Mutual_Funds",
    "Equity_Market",
    "Debentures",
    "Government_Bonds",
    "Fixed_Deposits",
    "PPF",
    "Gold"
]

# ======================================================
# SECTION 1: FINANCIAL HEALTH SNAPSHOT
# ======================================================
st.markdown("## 📊 Financial Health Snapshot")
st.divider()

# -----------------------------
# Currency Selection (FIRST)
# -----------------------------
currency_map = {
    "Indian Rupee (₹)": {"symbol": "₹", "to_inr": 1.0},
    "US Dollar ($)": {"symbol": "$", "to_inr": 83.0},
    "Euro (€)": {"symbol": "€", "to_inr": 90.0},
    "British Pound (£)": {"symbol": "£", "to_inr": 105.0}
}

cost_adjustment = {
    "Indian Rupee (₹)": 1.0,
    "US Dollar ($)": 1.25,
    "Euro (€)": 1.3,
    "British Pound (£)": 1.35
}

currency_choice = st.selectbox(
    "Select Currency",
    list(currency_map.keys())
)

currency_symbol = currency_map[currency_choice]["symbol"]
to_inr_rate = currency_map[currency_choice]["to_inr"]

st.caption("All inputs and outputs use the selected currency.")

# -----------------------------
# User Inputs (in chosen currency)
# -----------------------------
c1, c2, c3 = st.columns(3)

with c1:
    age = st.number_input("Age", 18, 100, 25)
    retirement_age = st.slider("Target Retirement Age", age + 1, 70, 60)

with c2:
    monthly_income = st.number_input(
        f"Monthly Income ({currency_symbol})",
        value=50000
    )
    tax_rate = st.slider("Tax Rate (%)", 0, 50, 10)

with c3:
    rent = st.number_input(f"Rent ({currency_symbol})", value=15000)
    food = st.number_input(f"Food ({currency_symbol})", value=8000)
    transport = st.number_input(f"Transport ({currency_symbol})", value=4000)
    others = st.number_input(f"Other Expenses ({currency_symbol})", value=3000)

# -----------------------------
# Convert to INR internally
# -----------------------------
income_inr = monthly_income * to_inr_rate
expenses_inr = (rent + food + transport + others) * to_inr_rate
tax_inr = income_inr * tax_rate / 100
savings_inr = income_inr - expenses_inr - tax_inr

# -----------------------------
# Convert back to display currency
# -----------------------------
expenses = expenses_inr / to_inr_rate
tax = tax_inr / to_inr_rate
savings = savings_inr / to_inr_rate

savings_rate = (savings / monthly_income) * 100 if monthly_income > 0 else 0
years_left = retirement_age - age

# -----------------------------
# Monthly Summary
# -----------------------------
st.markdown("### 💡 Monthly Summary")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Income", f"{currency_symbol}{monthly_income:,.2f}")
m2.metric("Expenses", f"{currency_symbol}{expenses:,.2f}")
m3.metric("Savings", f"{currency_symbol}{savings:,.2f}")
m4.metric("Years to Invest", f"{years_left} yrs")

# Savings Health Feedback
if savings_rate < 10:
    st.error("🚨 Low savings rate. Expenses are eating your future.")
elif savings_rate < 25:
    st.warning("⚠️ Moderate savings. Improvement possible.")
else:
    st.success("✅ Strong savings discipline.")

# -----------------------------
# Expense Breakdown Chart
# -----------------------------
expense_df = pd.DataFrame({
    "Category": ["Rent", "Food", "Transport", "Others"],
    "Amount": [rent, food, transport, others]
})

fig_exp = px.pie(
    expense_df,
    values="Amount",
    names="Category",
    hole=0.45,
    title="Expense Breakdown"
)

fig_exp.update_traces(textinfo="percent+label")

fig_exp.update_traces(pull=[0.03] * len(expense_df))

fig_exp.update_layout(
    showlegend=False,
    margin=dict(t=50, b=0, l=0, r=0)
)

st.plotly_chart(fig_exp, use_container_width=True)

# ======================================================
# SECTION 2: INVESTOR PROFILE & ML-BASED RECOMMENDATION
# ======================================================
st.header("🤖 Investor Profile & Recommendation")

st.markdown("**Rate your comfort with each investment (0 = Low, 5 = High)**")

# -----------------------------
# User Preference Inputs
# -----------------------------
r1, r2, r3, r4 = st.columns(4)
with r1:
    mf = st.slider("Mutual Funds", 0, 5, 3)
with r2:
    eq = st.slider("Equity", 0, 5, 3)
with r3:
    deb = st.slider("Debentures", 0, 5, 2)
with r4:
    bonds = st.slider("Govt Bonds", 0, 5, 2)

r5, r6, r7 = st.columns(3)
with r5:
    fd = st.slider("Fixed Deposits", 0, 5, 3)
with r6:
    ppf = st.slider("PPF", 0, 5, 2)
with r7:
    gold = st.slider("Gold", 0, 5, 2)

# -----------------------------
# ML: Cluster Prediction
# -----------------------------
user_features = np.array([[age, mf, eq, deb, bonds, fd, ppf, gold]])
user_scaled = recommender_scaler.transform(user_features)
cluster = kmeans_model.predict(user_scaled)[0]

cluster_profiles = {
    0: "Balanced investor with moderate risk appetite.",
    1: "Growth-oriented investor focused on higher returns.",
    2: "Conservative investor prioritizing capital protection.",
    3: "Long-term saver focused on stability."
}

st.success(f"🧠 **Investor Profile:** {cluster_profiles[cluster]}")

# ======================================================
# ML-BASED PORTFOLIO ALLOCATION (CENTROID-DRIVEN)
# ======================================================
st.subheader("📌Portfolio Allocation")

st.caption(
    "This allocation is generated using unsupervised machine learning. "
    "Your profile was matched to investors with similar behavior, and the "
    "cluster centroid represents the average portfolio composition."
)

# Extract centroid (excluding age)
centroid = kmeans_model.cluster_centers_[cluster][1:]

asset_names = [
    "Mutual Funds",
    "Equity",
    "Debentures",
    "Government Bonds",
    "Fixed Deposits",
    "PPF",
    "Gold"
]

weights = np.maximum(centroid, 0)
allocation_pct = (weights / weights.sum()) * 100

allocation_df = pd.DataFrame({
    "Asset Class": asset_names,
    "Allocation (%)": allocation_pct
})

st.dataframe(
    allocation_df.style.format({"Allocation (%)": "{:.2f}%"}),
    use_container_width=True
)

# ======================================================
# INVESTMENT AMOUNT ALLOCATION (ACTIONABLE OUTPUT)
# ======================================================
st.subheader("💼 Investment Amount Allocation")

investment_amount = st.number_input(
    f"Total Amount to Invest ({currency_symbol})",
    value=100000
)

allocation_df["Amount"] = (
    allocation_df["Allocation (%)"] / 100
) * investment_amount

st.dataframe(
    allocation_df.style.format({
        "Allocation (%)": "{:.2f}%",
        "Amount": f"{currency_symbol}{{:,.2f}}"
    }),
    use_container_width=True
)

# -----------------------------
# Allocation Visualization
# -----------------------------
fig_alloc = px.bar(
    allocation_df,
    x="Asset Class",
    y="Allocation (%)",
    title="Portfolio Allocation",
    text_auto=".2f",
    color="Asset Class"
)

fig_alloc.update_layout(
    yaxis_range=[0, 100],
    bargap=0.25,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)

fig_alloc.update_traces(
    marker=dict(
        line=dict(width=1, color="rgba(0,0,0,0.4)")
    ),
    textposition="outside"
)

st.plotly_chart(fig_alloc, use_container_width=True)

# Expected annual returns by asset class (industry-standard assumptions)
expected_returns = {
    "Equity": 0.12,
    "Mutual Funds": 0.10,
    "Debentures": 0.07,
    "Government Bonds": 0.06,
    "Fixed Deposits": 0.05,
    "PPF": 0.075,
    "Gold": 0.06
}

# ======================================================
# ML-ASSISTED MARKET INDEX RECOMMENDATION
# ======================================================
st.subheader("📊 Market Index Exposure")

st.caption(
    "Index recommendations are derived using your ML-inferred risk profile "
    "and historical market behavior. The model does not predict prices."
)

# Cluster → risk score (derived from centroid equity weight)
equity_weight = allocation_df.loc[
    allocation_df["Asset Class"] == "Equity", "Allocation (%)"
].values[0]

# Risk tolerance score (ML-driven, not hardcoded)
risk_score = equity_weight / 100  # 0–1

# Aggregate index behavior
index_summary = (
    index_df
    .groupby("Stock Index")
    .agg(
        avg_price=("Close Price", "mean"),
        volatility=("Close Price", "std")
    )
    .reset_index()
)

# Normalize metrics
index_summary["volatility_norm"] = (
    index_summary["volatility"] / index_summary["volatility"].max()
)

# ML-based suitability score
index_summary["Suitability_Score"] = (
    risk_score * (1 - index_summary["volatility_norm"])
)

# Rank indices
index_reco = index_summary.sort_values(
    "Suitability_Score", ascending=False
).head(3)

st.dataframe(
    index_reco[[
        "Stock Index",
        "avg_price",
        "volatility",
        "Suitability_Score"
    ]].style.format({
        "avg_price": "{:.2f}",
        "volatility": "{:.2f}",
        "Suitability_Score": "{:.3f}"
    }),
    use_container_width=True
)

st.plotly_chart(
    px.bar(
        index_reco,
        x="Stock Index",
        y="Suitability_Score",
        title="Index Suitability Ranking",
        text_auto=".3f"
    ),
    use_container_width=True
)

# ======================================================
# SECTION 3: ML-CONNECTED INVESTMENT GROWTH PROJECTION
# ======================================================
st.header("📈 Investment Growth Projection")

st.caption(
    "Portfolio growth is calculated by applying asset-specific growth rates "
    "to the portfolio allocation."
)

years = years_left

# Convert allocation % to usable form
alloc_map = dict(
    zip(allocation_df["Asset Class"], allocation_df["Allocation (%)"] / 100)
)

portfolio_growth = []
current_value = investment_amount

for year in range(1, years + 1):
    yearly_growth = 0

    for asset, weight in alloc_map.items():
        asset_return = expected_returns.get(asset, 0.05)
        yearly_growth += current_value * weight * asset_return

    current_value += yearly_growth
    portfolio_growth.append(current_value)

growth_df = pd.DataFrame({
    "Year": list(range(1, years + 1)),
    "Portfolio Value": portfolio_growth
})

fig_growth = px.line(
    growth_df,
    x="Year",
    y="Portfolio Value",
    title="Portfolio Growth Projection"
)

fig_growth.update_layout(
    yaxis_title=f"Portfolio Value ({currency_symbol})",
    xaxis_title="Year",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=50)
)

fig_growth.update_traces(line=dict(width=3))

st.plotly_chart(fig_growth, use_container_width=True)

st.caption("Tip: Long-term consistency matters more than short-term market movements.")

# ======================================================
# SECTION 4: FINANCIAL MILESTONES (CURRENCY-AWARE)
# ======================================================
st.header("🎯 Financial Milestones")

# Milestones defined in INR (base)
milestones_inr = {
    "Car Fund": 500_000,
    "House Down Payment": 2_000_000,
    "Retirement Corpus": 10_000_000
}

# Convert milestones to user currency
multiplier = cost_adjustment[currency_choice]

milestones_user_currency = {
    goal: round(((value * multiplier) / to_inr_rate) / 100) * 100
    for goal, value in milestones_inr.items()
}

final_value = portfolio_growth[-1]

for goal, target in milestones_user_currency.items():
    label = f"{goal} ({currency_symbol}{target:,.0f})"
    if final_value >= target:
        st.success(f"{label} → Achievable ✅")
    else:
        st.warning(f"{label} → Not yet achievable ⏳")

st.info(
    "All projections are estimates based on historical data, "
    "derived portfolio allocation, and assumed asset returns."
)