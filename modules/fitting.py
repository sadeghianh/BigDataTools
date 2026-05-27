# =========================
# modules/fitting.py
# Distribution Fitting and Central Limit Theorem module
# Fits data to theoretical distributions and
# demonstrates the Central Limit Theorem through simulation
# =========================

import pandas as pd                  # Import pandas for DataFrame operations
import numpy as np                   # Import numpy for numerical operations
import matplotlib.pyplot as plt      # Import matplotlib for plotting
import scipy.stats as sp             # Import scipy.stats for distribution fitting
import streamlit as st               # Import streamlit for UI rendering
from utils.helpers import (          # Import shared utilities
    get_numeric_columns,             # To list numeric columns
    drop_missing,                    # To clean NaN values
    section_header,                  # For styled section header
)


def render_fitting(df: pd.DataFrame):
    """
    Main function for the Distribution Fitting module.
    Allows the user to fit real data to theoretical distributions
    and run the Central Limit Theorem simulation.

    Parameters:
        df (pd.DataFrame): The uploaded dataset
    """
    section_header("Distribution Fitting & CLT", "🔗")  # Section heading

    # Sub-module selection: fitting or CLT simulation
    tab1, tab2 = st.tabs(["📐 Distribution Fitting", "📊 CLT Simulation"])  # Two tabs

    with tab1:
        _render_fitting_tab(df)   # Render the fitting UI

    with tab2:
        _render_clt_tab(df)       # Render the CLT simulation UI


def _render_fitting_tab(df: pd.DataFrame):
    """
    Render the Distribution Fitting sub-module.
    Fits selected data to multiple distributions and compares them.

    Parameters:
        df (pd.DataFrame): The uploaded dataset
    """
    st.markdown("### Fit Data to a Theoretical Distribution")  # Sub-heading

    # Get numeric columns for user to choose from
    numeric_cols = get_numeric_columns(df)
    if not numeric_cols:
        st.error("No numeric columns found.")
        return

    # Let user select which column's data to fit
    col = st.selectbox("Select column to fit:", numeric_cols, key="fit_col")

    # Let user select which distribution to fit
    dist_choice = st.selectbox(
        "Select distribution to fit:",
        ["Normal", "Exponential", "Poisson", "Uniform"],  # Available distributions
        key="fit_dist"
    )

    if st.button("Fit Distribution"):       # Trigger on button click
        series = drop_missing(df[col])      # Remove NaN values from the column

        # Need at least 10 data points for a meaningful fit
        if len(series) < 10:
            st.error("Need at least 10 data points to fit a distribution.")
            return

        # Call the appropriate fitting function based on user's choice
        if dist_choice == "Normal":
            _fit_normal(series, col)
        elif dist_choice == "Exponential":
            _fit_exponential(series, col)
        elif dist_choice == "Poisson":
            _fit_poisson(series, col)
        elif dist_choice == "Uniform":
            _fit_uniform(series, col)


def _fit_normal(series: pd.Series, col_name: str):
    """
    Fit a Normal distribution to the data and plot the result.
    Uses maximum likelihood estimation (MLE) via scipy.

    Parameters:
        series (pd.Series): Cleaned numeric data
        col_name (str): Column name for display
    """
    # sp.norm.fit returns (mean, std) estimated from the data
    mu, sigma = sp.norm.fit(series)

    # Create x values spanning the range of the data
    x = np.linspace(series.min(), series.max(), 300)

    # Compute the fitted PDF values at each x
    pdf_fitted = sp.norm.pdf(x, loc=mu, scale=sigma)

    # Create the comparison plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(series, bins=30, density=True,          # Actual data as histogram (normalized to density)
            alpha=0.5, color="#4C72B0", label="Data", edgecolor="white")
    ax.plot(x, pdf_fitted, "r-", linewidth=2,       # Fitted curve in red
            label=f"Normal fit: μ={mu:.2f}, σ={sigma:.2f}")
    ax.set_title(f"Normal Fit for '{col_name}'")    # Title
    ax.set_xlabel(col_name)
    ax.set_ylabel("Density")
    ax.legend()                                     # Show legend
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # Goodness-of-fit test using Kolmogorov-Smirnov test
    ks_stat, ks_p = sp.kstest(series, "norm", args=(mu, sigma))  # KS test against fitted normal

    # Show the fitted parameters and test results
    with st.expander("📋 Fitting Results", expanded=True):
        st.markdown("**Estimated Parameters:**")
        st.write(f"- Mean (μ): `{mu:.4f}`")
        st.write(f"- Std Dev (σ): `{sigma:.4f}`")
        st.write(f"- Variance (σ²): `{sigma**2:.4f}`")
        st.markdown("**Goodness-of-Fit (Kolmogorov-Smirnov Test):**")
        st.write(f"- KS Statistic: `{ks_stat:.4f}`")
        st.write(f"- P-value: `{ks_p:.4f}`")
        # Interpret the KS test result
        if ks_p > 0.05:
            st.success("✅ Data is consistent with a Normal distribution (p > 0.05).")
        else:
            st.warning("⚠️ Data does NOT fit a Normal distribution well (p ≤ 0.05).")
        # Compute and display additional statistics
        st.markdown("**Distribution Statistics:**")
        st.write(f"- Skewness: `{sp.skew(series):.4f}`")
        st.write(f"- Kurtosis (excess): `{sp.kurtosis(series):.4f}`")


def _fit_exponential(series: pd.Series, col_name: str):
    """
    Fit an Exponential distribution to the data.
    Only valid for non-negative data.

    Parameters:
        series (pd.Series): Cleaned numeric data
        col_name (str): Column name for display
    """
    # Exponential distribution requires non-negative values
    if series.min() < 0:
        st.error("Exponential distribution requires non-negative values.")
        return

    # sp.expon.fit returns (loc, scale) — here loc≈0, scale=1/λ
    loc, scale = sp.expon.fit(series)
    lam_est = 1 / scale          # Lambda = rate = 1/scale

    x = np.linspace(0, series.max(), 300)                # x from 0 to max
    pdf_fitted = sp.expon.pdf(x, loc=loc, scale=scale)   # Fitted PDF values

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(series, bins=30, density=True, alpha=0.5,
            color="#55A868", label="Data", edgecolor="white")
    ax.plot(x, pdf_fitted, "r-", linewidth=2,
            label=f"Exp fit: λ={lam_est:.3f}")
    ax.set_title(f"Exponential Fit for '{col_name}'")
    ax.set_xlabel(col_name)
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    ks_stat, ks_p = sp.kstest(series, "expon", args=(loc, scale))  # KS goodness-of-fit

    with st.expander("📋 Fitting Results", expanded=True):
        st.markdown("**Estimated Parameters:**")
        st.write(f"- Rate (λ): `{lam_est:.4f}`")
        st.write(f"- Mean (1/λ): `{1/lam_est:.4f}`")
        st.write(f"- Variance (1/λ²): `{1/lam_est**2:.4f}`")
        st.markdown("**Goodness-of-Fit:**")
        st.write(f"- KS Statistic: `{ks_stat:.4f}` | P-value: `{ks_p:.4f}`")
        if ks_p > 0.05:
            st.success("✅ Data is consistent with Exponential distribution (p > 0.05).")
        else:
            st.warning("⚠️ Poor fit to Exponential distribution (p ≤ 0.05).")
        st.write(f"- Skewness: `{sp.skew(series):.4f}` (Exponential theoretical: 2.0)")
        st.write(f"- Kurtosis: `{sp.kurtosis(series):.4f}` (Exponential theoretical: 6.0)")


def _fit_poisson(series: pd.Series, col_name: str):
    """
    Fit a Poisson distribution to count data using MLE (mean estimate).

    Parameters:
        series (pd.Series): Cleaned numeric data (should be non-negative integers)
        col_name (str): Column name for display
    """
    # For Poisson, the MLE estimate of λ is simply the sample mean
    lam_est = series.mean()   # Lambda = average count

    # x values are integer counts from 0 to max
    x = np.arange(0, int(series.max()) + 1)
    pmf_fitted = sp.poisson.pmf(x, mu=lam_est)  # Compute PMF values

    fig, ax = plt.subplots(figsize=(9, 5))
    # Compute relative frequency of each integer in the data
    value_counts = series.round().value_counts(normalize=True).sort_index()
    ax.bar(value_counts.index, value_counts.values,   # Actual frequencies
           alpha=0.5, color="#8172B2", label="Data", width=0.4, align="center")
    ax.plot(x, pmf_fitted, "ro-", markersize=5,       # Fitted PMF as connected dots
            linewidth=1.5, label=f"Poisson fit: λ={lam_est:.2f}")
    ax.set_title(f"Poisson Fit for '{col_name}'")
    ax.set_xlabel(col_name)
    ax.set_ylabel("Probability / Relative Frequency")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("📋 Fitting Results", expanded=True):
        st.markdown("**Estimated Parameters:**")
        st.write(f"- Lambda (λ): `{lam_est:.4f}`")
        st.write(f"- Mean = Variance = λ: `{lam_est:.4f}`")
        st.write(f"- Data Variance: `{series.var():.4f}` (should be ≈ λ for good fit)")
        st.write(f"- Skewness: `{sp.skew(series):.4f}` (Poisson theoretical: {1/np.sqrt(lam_est):.4f})")


def _fit_uniform(series: pd.Series, col_name: str):
    """
    Fit a Uniform distribution to the data using min and max as bounds.

    Parameters:
        series (pd.Series): Cleaned numeric data
        col_name (str): Column name for display
    """
    a = series.min()   # Lower bound estimate = minimum value
    b = series.max()   # Upper bound estimate = maximum value

    x = np.linspace(a - 0.1*(b-a), b + 0.1*(b-a), 300)  # x slightly wider than [a, b]
    pdf_fitted = sp.uniform.pdf(x, loc=a, scale=b-a)      # Fitted PDF (constant between a and b)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(series, bins=30, density=True, alpha=0.5,
            color="#CCB974", label="Data", edgecolor="white")
    ax.plot(x, pdf_fitted, "r-", linewidth=2,
            label=f"Uniform fit: [{a:.2f}, {b:.2f}]")
    ax.set_title(f"Uniform Fit for '{col_name}'")
    ax.set_xlabel(col_name)
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    ks_stat, ks_p = sp.kstest(series, "uniform", args=(a, b-a))  # KS goodness-of-fit

    with st.expander("📋 Fitting Results", expanded=True):
        st.write(f"- Lower bound (a): `{a:.4f}`")
        st.write(f"- Upper bound (b): `{b:.4f}`")
        st.write(f"- Mean: `{(a+b)/2:.4f}`")
        st.write(f"- Variance: `{(b-a)**2/12:.4f}`")
        st.write(f"- KS Statistic: `{ks_stat:.4f}` | P-value: `{ks_p:.4f}`")
        if ks_p > 0.05:
            st.success("✅ Data is consistent with Uniform distribution.")
        else:
            st.warning("⚠️ Poor fit to Uniform distribution.")


# =====================================================================
# Central Limit Theorem (CLT) Simulation
# =====================================================================

def _render_clt_tab(df: pd.DataFrame):
    """
    Render the Central Limit Theorem simulation.
    Takes many random samples, computes their means,
    and shows that the distribution of means is approximately Normal.

    Parameters:
        df (pd.DataFrame): The uploaded dataset
    """
    st.markdown("### Central Limit Theorem (CLT) Simulation")  # Sub-heading

    # Explanation box
    st.info("""
    **The Central Limit Theorem states:**
    If you take many random samples from *any* distribution, 
    the distribution of sample means will approach a Normal distribution,
    regardless of the original data's shape.
    """)

    # Get numeric columns to choose from
    numeric_cols = get_numeric_columns(df)
    if not numeric_cols:
        st.error("No numeric columns found.")
        return

    col = st.selectbox("Select column for CLT simulation:", numeric_cols, key="clt_col")

    # Let user configure the simulation parameters
    c1, c2 = st.columns(2)
    with c1:
        sample_size = st.slider(
            "Sample size (n) per draw:",
            min_value=2, max_value=500, value=30, key="clt_n"
        )  # How many values in each sample
    with c2:
        num_samples = st.slider(
            "Number of samples (repetitions):",
            min_value=50, max_value=5000, value=1000, step=50, key="clt_reps"
        )  # How many times to draw a sample

    if st.button("Run CLT Simulation"):              # Trigger simulation
        series = drop_missing(df[col])               # Remove NaN values

        if len(series) < sample_size:
            # Cannot draw a sample larger than the data
            st.error(f"Dataset has only {len(series)} values — reduce sample size below that.")
            return

        # Draw `num_samples` random samples, each of size `sample_size`
        # Compute the mean of each sample
        np.random.seed(42)   # Fix random seed for reproducibility
        sample_means = [
            np.mean(np.random.choice(series, size=sample_size, replace=True))
            # np.random.choice with replace=True allows bootstrap sampling
            for _ in range(num_samples)   # Repeat num_samples times
        ]
        sample_means = np.array(sample_means)   # Convert list to numpy array for operations

        # Create the CLT visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # Two subplots: original and sample means

        # LEFT: Original data distribution
        axes[0].hist(series, bins=30, density=True, alpha=0.7,
                     color="#4C72B0", edgecolor="white")
        axes[0].set_title(f"Original Data Distribution\n'{col}'")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Density")
        axes[0].grid(alpha=0.3)

        # RIGHT: Distribution of sample means
        axes[1].hist(sample_means, bins=40, density=True, alpha=0.7,
                     color="#55A868", edgecolor="white", label="Sample Means")

        # Overlay the theoretical normal curve predicted by CLT
        mu_clt = series.mean()                        # CLT predicted mean = population mean
        sigma_clt = series.std(ddof=1) / np.sqrt(sample_size)  # CLT predicted std = σ/√n
        x_norm = np.linspace(sample_means.min(), sample_means.max(), 300)  # x range for curve
        axes[1].plot(x_norm, sp.norm.pdf(x_norm, mu_clt, sigma_clt),      # Normal curve
                     "r-", linewidth=2.5, label="Theoretical Normal")
        axes[1].set_title(f"Distribution of {num_samples} Sample Means\n(n={sample_size} each)")
        axes[1].set_xlabel("Sample Mean")
        axes[1].set_ylabel("Density")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()     # Prevent label overlap
        st.pyplot(fig)         # Render plot
        plt.close(fig)         # Free memory

        # Show CLT statistics
        with st.expander("📋 CLT Simulation Statistics", expanded=True):
            st.markdown("**Original Population:**")
            st.write(f"- Mean (μ): `{series.mean():.4f}`")
            st.write(f"- Std Dev (σ): `{series.std(ddof=1):.4f}`")

            st.markdown("**Theoretical CLT Prediction for Sample Means:**")
            st.write(f"- Expected mean of means: `{mu_clt:.4f}`")
            st.write(f"- Expected std of means (σ/√n): `{sigma_clt:.4f}`")
            st.latex(r"\sigma_{\bar{x}} = \frac{\sigma}{\sqrt{n}}")  # CLT formula

            st.markdown("**Observed Sample Means Distribution:**")
            st.write(f"- Actual mean of means: `{sample_means.mean():.4f}`")
            st.write(f"- Actual std of means: `{sample_means.std():.4f}`")
            st.write(f"- Skewness of means: `{sp.skew(sample_means):.4f}` (should be ≈ 0)")
            st.write(f"- Kurtosis of means: `{sp.kurtosis(sample_means):.4f}` (should be ≈ 0)")

            st.success(
                f"✅ With n={sample_size}, the distribution of sample means is approximately Normal. "
                f"This demonstrates the Central Limit Theorem."
            )
