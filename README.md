# 📊 Statistical Data Dashboard

A professional, fully interactive statistical analysis dashboard built with Python and Streamlit.
Designed for university-level data science education and assignment presentations.

---

## 🚀 Features

| Module | Description |
|--------|-------------|
| **Statistics** | Mean, Median, Mode, Variance, Std Dev with formulas |
| **Plots** | Histogram, Boxplot, Scatter, Line, KDE, Spider, Gauge |
| **Sampling** | Random, Systematic, Stratified sampling |
| **Normalization** | Min-Max scaling, Z-score normalization |
| **Distributions** | Normal, Poisson, Exponential, Binomial, Bernoulli, Uniform |
| **Distribution Fitting** | Fit data, PMF/PDF/CDF, Skewness, Kurtosis |
| **CLT Simulation** | Central Limit Theorem demonstration |
| **Hypothesis Testing** | T-test, Z-test, ANOVA, Chi-square |

---

## 📁 Project Structure

```
data_dashboard/
│
├── app.py                  # Main Streamlit application entry point
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation
│
├── modules/
│   ├── stats.py            # Descriptive statistics functions
│   ├── plots.py            # All visualization functions
│   ├── sampling.py         # Sampling methods
│   ├── fitting.py          # Distribution fitting and CLT
│   ├── normalization.py    # Data normalization methods
│   ├── distributions.py    # Theoretical distributions
│   └── tests.py            # Hypothesis testing functions
│
└── utils/
    └── helpers.py          # Shared utility functions
```

---

## 🛠️ Installation & Run

### Step 1: Clone or download the project folder

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the dashboard
```bash
streamlit run app.py
```

### Step 4: Open in browser
Streamlit will automatically open `http://localhost:8501` in your browser.

---

## 📌 How to Use

1. **Upload** a CSV file using the sidebar uploader
2. **Navigate** using the sidebar menu to choose a module
3. **Select columns** and click buttons to see results
4. **Expand** result sections for detailed explanations

---

## 📚 Tech Stack

- **Python 3.10+**
- **Streamlit** — UI framework
- **Pandas** — Data manipulation
- **NumPy** — Numerical computing
- **SciPy** — Statistical tests and distributions
- **Matplotlib + Seaborn** — Static visualization
- **Plotly** — Interactive visualization
- **Statsmodels** — Advanced statistical models

---

## 👨‍🎓 Academic Use

This dashboard is designed for educational purposes. Every function is fully commented
with explanations of what each line does and why it is used.
