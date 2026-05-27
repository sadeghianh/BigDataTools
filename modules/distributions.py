# =========================
# modules/distributions.py
# Distributions module — uses REAL data from the uploaded dataset
# Fits parameters from actual data, plots real histogram vs theoretical curve
# If a distribution cannot be applied to the data, explains why
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as sp
import streamlit as st
from utils.helpers import get_numeric_columns, get_categorical_columns, drop_missing, section_header


def render_distributions(df=None):
    section_header("Probability Distributions", "🔔")

    # If no dataset is loaded, explain why we need one
    if df is None or (hasattr(df, 'empty') and df.empty):
        st.warning("⚠️ No dataset loaded. Please upload a CSV file to analyze real data.")
        return

    numeric_cols = get_numeric_columns(df)
    if not numeric_cols:
        st.error("No numeric columns found in the dataset.")
        return

    st.info(
        "📌 All parameters (μ, σ, λ, p, n) are **estimated from your real data** using "
        "Maximum Likelihood Estimation (MLE). The histogram shows your actual data; "
        "the curve shows the fitted theoretical distribution."
    )

    # Column selection
    col = st.selectbox("Select column to analyze:", numeric_cols, key="dist_col")
    series = drop_missing(df[col])

    if len(series) < 10:
        st.error("Need at least 10 data points for distribution analysis.")
        return

    # Distribution selection
    dist_type = st.selectbox(
        "Select distribution:",
        ["Normal", "Poisson", "Exponential", "Binomial", "Bernoulli", "Uniform"],
        key="dist_type_select"
    )

    st.markdown("---")

    if dist_type == "Normal":
        _render_normal(series, col)
    elif dist_type == "Poisson":
        _render_poisson(series, col)
    elif dist_type == "Exponential":
        _render_exponential(series, col)
    elif dist_type == "Binomial":
        _render_binomial(series, col, df)
    elif dist_type == "Bernoulli":
        _render_bernoulli(series, col, df)
    elif dist_type == "Uniform":
        _render_uniform(series, col)


# =====================================================================
# Distribution renderers — all use real data
# =====================================================================

def _render_normal(series: pd.Series, col: str):
    """
    Normal distribution fitted to real data using MLE.
    MLE for Normal: μ = sample mean, σ = sample std dev
    """
    st.markdown(f"### Normal Distribution fitted to '{col}'")

    # MLE parameter estimation: for Normal, MLE gives μ=mean, σ=std
    mu, sigma = sp.norm.fit(series)

    st.markdown(f"""
    **Parameters estimated from your data (MLE):**
    - μ (mean) = `{mu:.4f}` ← computed as: mean of '{col}'
    - σ (std dev) = `{sigma:.4f}` ← computed as: std dev of '{col}'
    """)

    x = np.linspace(series.min(), series.max(), 500)
    pdf = sp.norm.pdf(x, loc=mu, scale=sigma)
    cdf = sp.norm.cdf(x, loc=mu, scale=sigma)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # PDF: real histogram + fitted curve
    axes[0].hist(series, bins=30, density=True, alpha=0.5,
                 color="#4C72B0", edgecolor="white", label=f"Real data: {col}")
    axes[0].plot(x, pdf, color="#C44E52", linewidth=2.5, label="Fitted Normal PDF")
    axes[0].set_title(f"PDF — Real data vs Normal fit")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # CDF: empirical vs fitted
    sorted_data = np.sort(series)
    empirical_cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    axes[1].step(sorted_data, empirical_cdf, color="#4C72B0", linewidth=1.5, label="Empirical CDF")
    axes[1].plot(x, cdf, color="#C44E52", linewidth=2, linestyle="--", label="Fitted Normal CDF")
    axes[1].set_title("CDF — Empirical vs Fitted")
    axes[1].set_xlabel(col)
    axes[1].set_ylabel("F(x)  =  P(X ≤ x)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # Q-Q plot to check normality
    (osm, osr), (slope, intercept, r) = sp.probplot(series, dist="norm")
    axes[2].scatter(osm, osr, color="#4C72B0", s=10, alpha=0.6, label="Data quantiles")
    axes[2].plot(osm, slope*np.array(osm)+intercept, color="#C44E52", linewidth=2, label="Perfect normal line")
    axes[2].set_title("Q-Q Plot (normality check)")
    axes[2].set_xlabel("Theoretical quantiles")
    axes[2].set_ylabel(f"Sample quantiles of {col}")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Goodness-of-fit
    ks_stat, ks_p = sp.kstest(series, "norm", args=(mu, sigma))

    _show_dist_stats("Normal", {
        "n (sample size)": len(series),
        "μ — estimated mean": round(mu, 4),
        "σ — estimated std dev": round(sigma, 4),
        "σ² — variance": round(sigma**2, 4),
        "Skewness (data)": round(float(sp.skew(series)), 4),
        "Kurtosis excess (data)": round(float(sp.kurtosis(series)), 4),
        "KS statistic": round(ks_stat, 4),
        "KS p-value": round(ks_p, 4),
    })

    with st.expander("📖 Formula & Interpretation"):
        st.latex(r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}")
        st.latex(r"\hat{\mu}_{MLE} = \bar{x} = \frac{\sum x_i}{n}")
        st.latex(r"\hat{\sigma}_{MLE} = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n}}")
        if ks_p > 0.05:
            st.success(f"✅ KS test p={ks_p:.4f} > 0.05 → data is consistent with Normal distribution.")
        else:
            st.warning(f"⚠️ KS test p={ks_p:.4f} ≤ 0.05 → data does NOT fit Normal distribution well.")
        st.write("**Q-Q Plot**: if points fall on the red line → data is approximately normal.")


def _render_poisson(series: pd.Series, col: str):
    """
    Poisson distribution requires non-negative integer count data.
    Checks compatibility first, explains if not suitable.
    MLE for Poisson: λ = sample mean
    """
    st.markdown(f"### Poisson Distribution fitted to '{col}'")

    # Check compatibility: Poisson requires non-negative values
    if series.min() < 0:
        st.error(f"""
        ❌ **Poisson distribution cannot be applied to '{col}'.**

        **Reason:** Poisson models count data — it requires all values to be ≥ 0.
        Column '{col}' contains negative values (min = {series.min():.2f}).

        **What to do:** Use Poisson for columns like: number of events, frequency counts, etc.
        For '{col}', consider Normal or Exponential distribution instead.
        """)
        return

    # Warn if data looks non-integer (continuous data)
    non_integer_pct = (series != series.round()).mean() * 100
    if non_integer_pct > 10:
        st.warning(f"""
        ⚠️ **Caution:** {non_integer_pct:.1f}% of values in '{col}' are not integers.

        Poisson is designed for **discrete count data** (0, 1, 2, 3, ...).
        For continuous data like '{col}', the fit may be poor.
        Proceeding anyway — check the KS p-value for fit quality.
        """)

    # MLE for Poisson: λ = sample mean
    lam = series.mean()
    st.markdown(f"""
    **Parameter estimated from your data (MLE):**
    - λ = `{lam:.4f}` ← computed as: mean of '{col}' = {series.mean():.4f}
    - For Poisson: MLE gives λ̂ = x̄ (sample mean)
    """)
    st.latex(r"\hat{\lambda}_{MLE} = \bar{x} = \frac{\sum x_i}{n}")

    x_int = np.arange(0, int(series.max()) + 2)
    pmf = sp.poisson.pmf(x_int, mu=lam)
    cdf_vals = sp.poisson.cdf(x_int, mu=lam)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Empirical frequency vs fitted PMF
    val_counts = series.round().value_counts(normalize=True).sort_index()
    axes[0].bar(val_counts.index, val_counts.values, alpha=0.5,
                color="#4C72B0", label=f"Real data: {col}", width=0.4)
    axes[0].plot(x_int, pmf, "ro-", markersize=5, linewidth=1.5,
                 label=f"Fitted Poisson PMF (λ={lam:.2f})")
    axes[0].set_title(f"PMF — Real data vs Poisson fit")
    axes[0].set_xlabel(f"{col}  (k = count)")
    axes[0].set_ylabel("P(X = k)  /  Relative frequency")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # Empirical CDF vs fitted CDF
    sorted_data = np.sort(series)
    empirical_cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    axes[1].step(sorted_data, empirical_cdf, color="#4C72B0", linewidth=1.5, label="Empirical CDF")
    axes[1].step(x_int, cdf_vals, color="#C44E52", linewidth=2, linestyle="--",
                 label=f"Fitted Poisson CDF")
    axes[1].set_title("CDF — Empirical vs Fitted")
    axes[1].set_xlabel(col)
    axes[1].set_ylabel("F(x)  =  P(X ≤ x)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    ks_stat, ks_p = sp.kstest(series, lambda x: sp.poisson.cdf(x, mu=lam))

    _show_dist_stats("Poisson", {
        "n (sample size)": len(series),
        "λ — estimated rate (= mean)": round(lam, 4),
        "Data mean": round(float(series.mean()), 4),
        "Data variance (should ≈ λ)": round(float(series.var()), 4),
        "Skewness (theoretical: 1/√λ)": round(1/np.sqrt(lam), 4),
        "Skewness (data)": round(float(sp.skew(series)), 4),
        "KS statistic": round(ks_stat, 4),
        "KS p-value": round(ks_p, 4),
    })

    with st.expander("📖 Formula & Interpretation"):
        st.latex(r"P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}")
        if ks_p > 0.05:
            st.success(f"✅ KS p={ks_p:.4f} > 0.05 → data is consistent with Poisson.")
        else:
            st.warning(f"⚠️ KS p={ks_p:.4f} ≤ 0.05 → poor fit. Check if data variance ≈ mean.")
        st.write("**Key check for Poisson:** mean ≈ variance. If they differ a lot, Poisson is not the right model.")


def _render_exponential(series: pd.Series, col: str):
    """
    Exponential distribution requires non-negative data.
    MLE for Exponential: λ = 1 / sample mean
    """
    st.markdown(f"### Exponential Distribution fitted to '{col}'")

    if series.min() < 0:
        st.error(f"""
        ❌ **Exponential distribution cannot be applied to '{col}'.**

        **Reason:** Exponential models time/duration between events — requires all values ≥ 0.
        Column '{col}' has negative values (min = {series.min():.2f}).

        **What to do:** Use Exponential for columns representing durations, waiting times, or gaps.
        For '{col}', try Normal distribution instead.
        """)
        return

    # MLE for Exponential: loc=0, scale=1/λ=mean
    loc, scale = sp.expon.fit(series, floc=0)   # floc=0 forces location=0 (standard exponential)
    lam = 1 / scale

    st.markdown(f"""
    **Parameters estimated from your data (MLE):**
    - λ (rate) = `{lam:.6f}` ← computed as: 1 / mean = 1 / {series.mean():.4f}
    - Mean = 1/λ = `{scale:.4f}`
    """)
    st.latex(r"\hat{\lambda}_{MLE} = \frac{1}{\bar{x}}")

    x = np.linspace(0, series.max(), 500)
    pdf = sp.expon.pdf(x, loc=loc, scale=scale)
    cdf_vals = sp.expon.cdf(x, loc=loc, scale=scale)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].hist(series, bins=30, density=True, alpha=0.5,
                 color="#55A868", edgecolor="white", label=f"Real data: {col}")
    axes[0].plot(x, pdf, color="#C44E52", linewidth=2.5,
                 label=f"Fitted Exp PDF (λ={lam:.4f})")
    axes[0].set_title("PDF — Real data vs Exponential fit")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    sorted_data = np.sort(series)
    emp_cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    axes[1].step(sorted_data, emp_cdf, color="#55A868", linewidth=1.5, label="Empirical CDF")
    axes[1].plot(x, cdf_vals, color="#C44E52", linewidth=2, linestyle="--",
                 label="Fitted Exponential CDF")
    axes[1].set_title("CDF — Empirical vs Fitted")
    axes[1].set_xlabel(col)
    axes[1].set_ylabel("F(x)  =  P(X ≤ x)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    ks_stat, ks_p = sp.kstest(series, "expon", args=(loc, scale))

    _show_dist_stats("Exponential", {
        "n (sample size)": len(series),
        "λ — estimated rate": round(lam, 6),
        "Mean (1/λ)": round(scale, 4),
        "Variance (1/λ²)": round(scale**2, 4),
        "Skewness (theoretical: 2)": 2,
        "Skewness (data)": round(float(sp.skew(series)), 4),
        "Kurtosis excess (theoretical: 6)": 6,
        "Kurtosis excess (data)": round(float(sp.kurtosis(series)), 4),
        "KS statistic": round(ks_stat, 4),
        "KS p-value": round(ks_p, 4),
    })

    with st.expander("📖 Formula & Interpretation"):
        st.latex(r"f(x) = \lambda e^{-\lambda x}, \quad x \geq 0")
        st.latex(r"\hat{\lambda}_{MLE} = \frac{1}{\bar{x}} = \frac{n}{\sum x_i}")
        if ks_p > 0.05:
            st.success(f"✅ KS p={ks_p:.4f} > 0.05 → data is consistent with Exponential.")
        else:
            st.warning(f"⚠️ KS p={ks_p:.4f} ≤ 0.05 → poor fit. Exponential expects skewness≈2.")


def _render_binomial(series: pd.Series, col: str, df: pd.DataFrame):
    """
    Binomial requires knowing n (number of trials).
    Estimates p from data. Checks compatibility.
    """
    st.markdown(f"### Binomial Distribution fitted to '{col}'")

    # Check: values must be non-negative integers
    if series.min() < 0:
        st.error(f"""
        ❌ **Binomial distribution cannot be applied to '{col}'.**

        **Reason:** Binomial counts successes in n trials — requires values ≥ 0.
        Column '{col}' has negative values (min = {series.min():.2f}).
        """)
        return

    non_int_pct = (series != series.round()).mean() * 100
    if non_int_pct > 5:
        st.warning(f"""
        ⚠️ **{non_int_pct:.1f}% of values in '{col}' are not integers.**

        Binomial counts discrete successes (0, 1, 2, ..., n).
        Column '{col}' appears to be continuous data.
        Proceeding with rounded values — fit quality may be poor.
        """)
        series = series.round().astype(int)

    # n must be specified by user — it's the number of trials per observation
    n_max = int(series.max())
    st.info(f"""
    ℹ️ **Binomial requires knowing n (number of trials per observation).**

    The max value in '{col}' is {n_max} — this is used as the upper bound.
    You can adjust n below if you know the actual number of trials.
    """)
    n = st.slider("n (number of trials per observation):", n_max, n_max*3, n_max, 1, key="binom_n_real")

    # MLE for Binomial: p̂ = mean / n
    p_hat = series.mean() / n
    p_hat = np.clip(p_hat, 0.001, 0.999)   # Clip to valid range

    st.markdown(f"""
    **Parameters estimated from your data (MLE):**
    - n = `{n}` (number of trials — set above)
    - p̂ = mean / n = `{series.mean():.4f}` / `{n}` = `{p_hat:.4f}`
    """)
    st.latex(r"\hat{p}_{MLE} = \frac{\bar{x}}{n}")

    x_vals = np.arange(0, n + 1)
    pmf = sp.binom.pmf(x_vals, n=n, p=p_hat)
    cdf_vals = sp.binom.cdf(x_vals, n=n, p=p_hat)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    val_counts = series.value_counts(normalize=True).sort_index()
    axes[0].bar(val_counts.index, val_counts.values, alpha=0.5,
                color="#4C72B0", label=f"Real data: {col}", width=0.4)
    axes[0].plot(x_vals, pmf, "ro-", markersize=4, linewidth=1.5,
                 label=f"Fitted B(n={n}, p={p_hat:.3f})")
    axes[0].set_title("PMF — Real data vs Binomial fit")
    axes[0].set_xlabel(f"{col}  (k = successes)")
    axes[0].set_ylabel("P(X = k)  /  Relative frequency")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    sorted_data = np.sort(series)
    emp_cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    axes[1].step(sorted_data, emp_cdf, color="#4C72B0", linewidth=1.5, label="Empirical CDF")
    axes[1].step(x_vals, cdf_vals, color="#C44E52", linewidth=2, linestyle="--",
                 label="Fitted Binomial CDF")
    axes[1].set_title("CDF — Empirical vs Fitted")
    axes[1].set_xlabel(col)
    axes[1].set_ylabel("F(x)  =  P(X ≤ x)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    _show_dist_stats("Binomial", {
        "n (trials)": n,
        "p̂ — estimated probability": round(p_hat, 4),
        "n (sample size)": len(series),
        "Mean (np)": round(n * p_hat, 4),
        "Variance (np(1-p))": round(n * p_hat * (1 - p_hat), 4),
        "Skewness (data)": round(float(sp.skew(series)), 4),
        "Kurtosis excess (data)": round(float(sp.kurtosis(series)), 4),
    })

    with st.expander("📖 Formula & Interpretation"):
        st.latex(r"P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}")
        st.latex(r"\hat{p}_{MLE} = \frac{\bar{x}}{n}")


def _render_bernoulli(series: pd.Series, col: str, df: pd.DataFrame):
    """
    Bernoulli requires binary data (only 0 and 1).
    Checks if column is binary. If not, suggests a categorical column or explains why.
    """
    st.markdown(f"### Bernoulli Distribution fitted to '{col}'")

    unique_vals = series.dropna().unique()

    # Check if column is binary (only 0 and 1)
    is_binary_numeric = set(unique_vals).issubset({0, 1, 0.0, 1.0})

    if not is_binary_numeric:
        st.error(f"""
        ❌ **Bernoulli distribution cannot be directly applied to '{col}'.**

        **Reason:** Bernoulli models a single trial with exactly two outcomes: 0 (failure) or 1 (success).
        Column '{col}' has {len(unique_vals)} unique values: {sorted(unique_vals[:5])}{' ...' if len(unique_vals)>5 else ''} — not binary.

        **What you can do instead:**
        """)

        # Suggest categorical columns as alternatives
        cat_cols = get_categorical_columns(df)
        if cat_cols:
            st.info(f"""
            👉 **Option A — Use a categorical column:**
            Your dataset has these categorical columns: {cat_cols}
            Select one of them below to convert to 0/1 and apply Bernoulli.
            """)
            cat_col = st.selectbox("Select categorical column:", cat_cols, key="bern_cat_alt")
            cat_series = df[cat_col].dropna()
            cat_unique = cat_series.unique().tolist()
            if len(cat_unique) == 2:
                success_val = st.selectbox(f"Which value counts as SUCCESS (1)?", cat_unique, key="bern_success")
                binary = (cat_series == success_val).astype(int)
                st.success(f"✅ Converted '{cat_col}': '{success_val}'=1, other=0. Now applying Bernoulli.")
                _apply_bernoulli(binary, f"{cat_col} (binary: {success_val}=1)")
            else:
                st.warning(f"Column '{cat_col}' has {len(cat_unique)} values — need exactly 2 for Bernoulli.")

        st.info("""
        👉 **Option B — Threshold numeric column:**
        Convert '{col}' to binary by choosing a threshold (e.g. Salary > 55000 = 1, else 0).
        """)
        threshold = st.number_input(
            f"Threshold: '{col}' > threshold → 1, else → 0",
            value=float(series.median()), key="bern_threshold"
        )
        binary_threshold = (series > threshold).astype(int)
        n_ones = binary_threshold.sum()
        n_zeros = len(binary_threshold) - n_ones
        st.write(f"Result: {n_ones} ones ({100*n_ones/len(binary_threshold):.1f}%), {n_zeros} zeros")
        if st.button("Apply Bernoulli with this threshold"):
            _apply_bernoulli(binary_threshold, f"{col} > {threshold:.2f}")
        return

    # If already binary, apply directly
    _apply_bernoulli(series.astype(int), col)


def _apply_bernoulli(binary: pd.Series, label: str):
    """Apply Bernoulli distribution to a binary (0/1) series."""
    # MLE for Bernoulli: p̂ = proportion of successes
    p_hat = binary.mean()

    st.markdown(f"""
    **Parameter estimated from your data (MLE):**
    - p̂ = proportion of 1s = `{binary.sum()}` / `{len(binary)}` = `{p_hat:.4f}`
    """)
    st.latex(r"\hat{p}_{MLE} = \frac{\text{number of successes}}{n} = \bar{x}")

    x = np.array([0, 1])
    pmf = sp.bernoulli.pmf(x, p=p_hat)
    cdf_vals = sp.bernoulli.cdf(x, p=p_hat)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Real counts vs fitted PMF
    real_counts = binary.value_counts(normalize=True).reindex([0, 1], fill_value=0)
    axes[0].bar(["Failure (0)", "Success (1)"], real_counts.values,
                color=["#C44E52", "#55A868"], edgecolor="white", alpha=0.6,
                label=f"Real data: {label}")
    axes[0].plot(["Failure (0)", "Success (1)"], pmf, "ko--", markersize=8,
                 linewidth=1.5, label=f"Fitted Ber(p={p_hat:.3f})")
    axes[0].set_title(f"PMF — Real data vs Bernoulli fit")
    axes[0].set_ylabel("Probability")
    axes[0].set_ylim(0, 1.1)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].step([0, 1, 2], [0] + list(cdf_vals), where="post",
                 color="#C44E52", linewidth=2, label="Fitted Bernoulli CDF")
    axes[1].set_title("CDF")
    axes[1].set_xlabel("x  (0=failure, 1=success)")
    axes[1].set_ylabel("F(x)  =  P(X ≤ x)")
    axes[1].set_xticks([0, 1])
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    _show_dist_stats("Bernoulli", {
        "n (sample size)": len(binary),
        "p̂ — estimated success prob": round(p_hat, 4),
        "Count of 1s (successes)": int(binary.sum()),
        "Count of 0s (failures)": int((binary == 0).sum()),
        "Mean (= p̂)": round(p_hat, 4),
        "Variance (p(1-p))": round(p_hat * (1 - p_hat), 4),
        "Skewness (theoretical)": round((1-2*p_hat)/np.sqrt(p_hat*(1-p_hat)), 4) if 0 < p_hat < 1 else 0,
    })

    with st.expander("📖 Formula & Interpretation"):
        st.latex(r"P(X = k) = p^k (1-p)^{1-k}, \quad k \in \{0, 1\}")
        st.latex(r"\hat{p}_{MLE} = \bar{x} = \frac{\sum x_i}{n}")


def _render_uniform(series: pd.Series, col: str):
    """
    Uniform distribution: parameters estimated from data.
    MLE for Uniform: a = min, b = max
    Note: any continuous data can be compared to Uniform — we explain fit quality.
    """
    st.markdown(f"### Uniform Distribution fitted to '{col}'")

    # MLE for Uniform: a=min, b=max
    a = series.min()
    b = series.max()

    st.markdown(f"""
    **Parameters estimated from your data (MLE):**
    - a (lower bound) = `{a:.4f}` ← min of '{col}'
    - b (upper bound) = `{b:.4f}` ← max of '{col}'

    **Note:** For Uniform distribution, MLE gives a = min(x), b = max(x).
    A Uniform distribution assumes all values in [a, b] are equally likely.
    If your data is concentrated in the middle, the fit will be poor (expected).
    """)
    st.latex(r"\hat{a}_{MLE} = \min(x_i), \quad \hat{b}_{MLE} = \max(x_i)")

    x = np.linspace(a - 0.05*(b-a), b + 0.05*(b-a), 500)
    pdf = sp.uniform.pdf(x, loc=a, scale=b-a)
    cdf_vals = sp.uniform.cdf(x, loc=a, scale=b-a)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].hist(series, bins=30, density=True, alpha=0.5,
                 color="#CCB974", edgecolor="white", label=f"Real data: {col}")
    axes[0].plot(x, pdf, color="#C44E52", linewidth=2.5,
                 label=f"Fitted Uniform PDF\n[{a:.2f}, {b:.2f}]")
    axes[0].set_title("PDF — Real data vs Uniform fit")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    sorted_data = np.sort(series)
    emp_cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    axes[1].step(sorted_data, emp_cdf, color="#CCB974", linewidth=1.5, label="Empirical CDF")
    axes[1].plot(x, cdf_vals, color="#C44E52", linewidth=2, linestyle="--",
                 label="Fitted Uniform CDF")
    axes[1].set_title("CDF — Empirical vs Fitted")
    axes[1].set_xlabel(col)
    axes[1].set_ylabel("F(x)  =  P(X ≤ x)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    ks_stat, ks_p = sp.kstest(series, "uniform", args=(a, b-a))

    _show_dist_stats("Uniform", {
        "n (sample size)": len(series),
        "a — lower bound (min)": round(a, 4),
        "b — upper bound (max)": round(b, 4),
        "Mean ((a+b)/2)": round((a+b)/2, 4),
        "Variance ((b-a)²/12)": round((b-a)**2/12, 4),
        "Skewness (theoretical: 0)": 0,
        "Skewness (data)": round(float(sp.skew(series)), 4),
        "Kurtosis excess (theoretical: -1.2)": -1.2,
        "Kurtosis excess (data)": round(float(sp.kurtosis(series)), 4),
        "KS statistic": round(ks_stat, 4),
        "KS p-value": round(ks_p, 4),
    })

    with st.expander("📖 Formula & Interpretation"):
        st.latex(r"f(x) = \frac{1}{b - a}, \quad a \leq x \leq b")
        st.latex(r"\hat{a}_{MLE} = \min(x_i), \quad \hat{b}_{MLE} = \max(x_i)")
        if ks_p > 0.05:
            st.success(f"✅ KS p={ks_p:.4f} > 0.05 → data is consistent with Uniform distribution.")
        else:
            st.warning(f"⚠️ KS p={ks_p:.4f} ≤ 0.05 → data does not fit Uniform well (common for real data).")
        st.write("Most real-world data is NOT uniform. If KS fails, it simply means the data has a non-uniform shape.")


def _show_dist_stats(dist_name: str, stats: dict):
    """Display distribution statistics table."""
    st.markdown(f"#### 📋 {dist_name} — Estimated Parameters & Statistics")
    stats_df = pd.DataFrame({"Property": list(stats.keys()), "Value": list(stats.values())})
    stats_df["Value"] = stats_df["Value"].apply(
        lambda v: round(v, 4) if isinstance(v, float) else v
    )
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
