# 📊 Statistical Analysis Dashboard

**SRH University · Tools & Methods of Data Analysis**

A user-friendly, interactive Data Analysis Toolkit built with Python and Streamlit.
It lets anyone — even with **no programming experience** — upload a dataset and run a
complete statistical analysis through a clean point-and-click interface.

### 🔗 Live App
**https://srh-stats-dashboard.streamlit.app**

> Just open the link, upload a CSV file, and start exploring. Nothing to install.

---

## ✨ Toolkit Features

The dashboard is organized into simple modules, each accessible from the sidebar menu:

| Module | What it does |
|--------|--------------|
| **🏠 Home** | Upload your data, preview it, and set units for your columns |
| **🔎 Data Profile** | "Know your data": row/column counts, missing values, normality per column, plus **data cleaning & export** |
| **📐 Descriptive Statistics** | Mean, median, mode, variance, standard deviation — with formulas and plain explanations |
| **📈 Visualizations** | Histogram, Box plot, Scatter, KDE, Violin, Bar, and Line charts (interactive) |
| **🎲 Sampling** | Random, Systematic, and Stratified sampling — with downloadable samples |
| **⚖️ Normalization** | Min-Max scaling and Z-score standardization, with before/after charts |
| **🔔 Distributions** | Fit your data to Normal, Poisson, Exponential, Binomial, Bernoulli, Uniform — see PDFs & CDFs |
| **🔗 Fitting & CLT** | Distribution fitting with goodness-of-fit + a Central Limit Theorem simulation |
| **🧪 Hypothesis Testing** | A full suite of tests, grouped by type (see below) + a **Test Advisor** |
| **🎯 Confidence Intervals** | Estimate the range that contains the true population mean |

### Statistical tests included
- **Parametric:** One-Sample t-test, Two-Sample t-test, Z-test, One-Way ANOVA, Two-Way ANOVA
- **Non-Parametric:** Mann-Whitney U, Wilcoxon Signed-Rank, Kruskal-Wallis, Fisher's Exact, Chi-Square
- **Correlation:** Pearson correlation (with significance test and scatter + regression line)
- **Normality:** Shapiro-Wilk and D'Agostino tests, with histogram and Q-Q plot
- **🧭 Test Advisor:** Answer a few questions about your data and goal, and the dashboard
  examines your dataset (sample size, normality, number of groups) and **recommends the
  right test** with reasons.

---

## 🚀 How to Use It

1. **Open the app** at the link above.
2. **Upload a CSV** using the **"Upload a CSV file"** box in the sidebar.
   (A sample dataset is provided in this repository — see below.)
3. **(Optional) Set units** for your numeric columns on the Home page
   (e.g. `$` for salary, `years` for age). These appear on all chart axes.
4. **Pick a module** from the sidebar menu and follow the on-screen options.
   Every screen explains what it does and interprets the results for you.

### Tip for first-time users
Start with **🔎 Data Profile** — it tells you how big your data is, whether it's
normally distributed, and what kind of tests are appropriate. Then visit the
**🧭 Test Advisor** (inside Hypothesis Testing) if you're unsure which test to use.

---

## 📈 Example Analysis (using the sample IBM HR dataset)

1. Upload `IBM_HR_Analytics.csv`.
2. Set the unit of `MonthlyIncome` to `$` on the Home page.
3. Go to **🧪 Hypothesis Testing → Non-Parametric → Chi-Square Test**.
   Choose `Attrition` and `OverTime`.
   → Result: a very strong association (χ² ≈ 85, p < 0.0001) — employees who work
   overtime leave the company about **4× more often**.
4. Go to **🎯 Confidence Intervals**, choose `MonthlyIncome`, 95%.
   → Result: a range that very likely contains the true average salary.
5. Go to **📐 Descriptive Statistics** and **📈 Visualizations** to summarize and
   plot the data.

A full written analysis (Word report) and a presentation of this example are included
in the project deliverables.

---

## 📂 Sample Dataset

This repository includes a sample dataset for demonstration:

- **`IBM_HR_Analytics.csv`** — 1,470 employee records with 25 variables
  (age, salary, department, overtime, attrition, satisfaction scores, etc.).
  Based on the public *IBM HR Analytics Employee Attrition* dataset from Kaggle.

You can also upload **any** CSV file of your own — the toolkit adapts automatically.

---

## 🛠️ Running It Locally (optional, for developers)

```bash
# 1. Clone the repository
git clone https://github.com/sadeghianh/BigDataTools.git
cd BigDataTools

# 2. Install the dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## 🧱 Project Structure

```
BigDataTools/
├── app.py                  # Main application (run this)
├── requirements.txt        # Python dependencies
├── IBM_HR_Analytics.csv    # Sample dataset
├── modules/                # One file per feature
│   ├── data_profile.py     # Know-your-data + cleaning & export
│   ├── stats.py            # Descriptive statistics
│   ├── plots.py            # Visualizations
│   ├── sampling.py         # Sampling methods
│   ├── normalization.py    # Scaling / standardization
│   ├── distributions.py    # Probability distributions (PDF/CDF)
│   ├── fitting.py          # Distribution fitting + CLT
│   ├── tests.py            # Hypothesis tests (parametric + correlation)
│   ├── nonparametric.py    # Non-parametric tests + Test Advisor
│   └── confidence.py       # Confidence intervals
└── utils/
    └── helpers.py          # Shared helper functions (units, formatting, logo)
```

---

## 🎨 Design Notes

- **Easy to use:** point-and-click interface, no coding required.
- **Well documented:** every line of code is commented; every result is explained in plain language.
- **Visually appealing:** clean modern theme with the SRH University colors.
- **Organized & modular:** each statistical method lives in its own file.
- **Accessible:** built-in explanations, a Test Advisor, and unit labels make it approachable
  for users with limited statistics or programming background.

---

*Built with Python, Streamlit, pandas, NumPy, SciPy, statsmodels, matplotlib, seaborn, and Plotly.*
