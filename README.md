# 💰 AI Financial Advisor
### Machine Learning–Driven Personal Finance & Investment Recommendation System

🔗 **Live Application:**  
https://sank-hub-ai-financial-advisor-app-mcpqzc.streamlit.app/

---

## 📖 Overview
The **AI Financial Advisor** is an end-to-end machine learning–powered web application that analyzes a user’s financial situation, infers their investment behavior, and generates **data-driven portfolio recommendations**.

Unlike rule-based finance apps, this system uses **unsupervised machine learning and behavioral clustering** to produce personalized investment insights.

---

## 🧠 Machine Learning Methodology (Core Focus)

### 1️⃣ Investor Profiling (Unsupervised ML)
- **Algorithm:** KMeans Clustering  
- **Features Used:**  
  - Age  
  - Mutual Funds  
  - Equity Market  
  - Debentures  
  - Government Bonds  
  - Fixed Deposits  
  - PPF  
  - Gold  

Each user is mapped to an **investor cluster** based on similarity to historical investor behavior.

---

### 2️⃣ ML-Derived Portfolio Allocation
- Portfolio weights are **not manually defined**
- Allocation is derived from the **cluster centroid**
- Represents the *average investment behavior* of similar investors

➡️ This ensures the recommendation is **purely ML-driven**, not hardcoded.

---

### 3️⃣ Risk-Aware Market Index Recommendation
- Uses historical market data (Dow Jones, S&P 500, etc.)
- Volatility is calculated from price movements
- User’s **ML-inferred risk tolerance** determines index suitability
- Outputs **top market indices ranked by ML suitability score**

---

### 4️⃣ ML-Connected Growth Projection
- Portfolio growth is simulated using:
  - ML-derived allocation
  - Asset-specific expected returns
- Produces a **long-term investment projection**
- Dynamically adapts to:
  - User age
  - Retirement horizon
  - Risk profile

---

## 🌍 Currency-Aware & Region-Adjusted
- Supports:
  - INR (₹)
  - USD ($)
  - EUR (€)
  - GBP (£)
- Automatically adjusts:
  - Investment values
  - Financial milestones
  - Cost-of-living differences

---

## 🎯 Financial Milestones
The system evaluates whether the user can realistically achieve:
- 🚗 Car Fund
- 🏠 House Down Payment
- 🧓 Retirement Corpus  

All milestones are **currency-adjusted and ML-projection based**.

---

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Machine Learning:** Scikit-learn (KMeans)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly
- **Model Persistence:** Joblib
- **Deployment:** Streamlit Cloud

