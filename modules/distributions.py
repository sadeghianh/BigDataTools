# =========================
# modules/distributions.py
# Distributions module — uses REAL data from the uploaded dataset
# Fits parameters from actual data, plots real histogram vs theoretical curve
# If a distribution cannot be applied to the data, explains why
# =========================

# Import numpy for numerical array operations and math functions
import numpy as np  # Import numpy for numerical array operations and math functions
# Import pandas for DataFrame and data manipulation
import pandas as pd  # Import pandas for DataFrame and data manipulation
# Import matplotlib for creating static plots and charts
import matplotlib.pyplot as plt  # Import matplotlib for creating static plots
# Import gridspec for advanced subplot layout control
import matplotlib.gridspec as gridspec  # Import gridspec for advanced subplot layout control
# Import scipy.stats for statistical tests and distributions
from scipy import stats as sp  # Import scipy.stats for statistical distributions and tests
# Import streamlit to build the interactive web UI
import streamlit as st  # Import streamlit to build the interactive web UI
# Import utils library
from utils.helpers import (              # Import shared utility functions
    get_numeric_columns,                 # To list numeric columns
    get_categorical_columns,             # To list categorical columns
    drop_missing,                        # To remove NaN values
    section_header,                      # For styled section header
    label_with_unit,                     # Build axis label with unit
    density_label,                       # Standard density Y-axis label
    probability_label,                   # Standard PMF Y-axis label
    cdf_label,                           # Standard CDF Y-axis label
    render_inline_unit_input,            # Inline unit input widget
)


# Define the render_distributions function
def render_distributions(df=None):  # Define the render_distributions function
    # Execute this operation
    section_header("Probability Distributions", "🔔")  # Execute this statement

    # If no dataset is loaded, explain why we need one
    # Check condition and branch accordingly
    if df is None or (hasattr(df, 'empty') and df.empty):  # Check condition
        # Show a yellow warning message to the user
        st.warning("⚠️ No dataset loaded. Please upload a CSV file to analyze real data.")  # Show a yellow warning message
        # Exit the function early — stop execution here
        return  # Exit function early

    # Pandas data operation
    numeric_cols = get_numeric_columns(df)  # Store result in numeric_cols
    # Check condition and branch accordingly
    if not numeric_cols:  # Check condition
        # Show a red error message to the user
        st.error("No numeric columns found in the dataset.")  # Show a red error message
        # Exit the function early — stop execution here
        return  # Exit function early

    # Show an informational blue message box
    st.info(  # Show an informational blue message box
        # Execute this operation
        "📌 All parameters (μ, σ, λ, p, n) are **estimated from your real data** using "  # Execute this statement
        # Execute this operation
        "Maximum Likelihood Estimation (MLE). The histogram shows your actual data; "  # Execute this statement
        # Execute this operation
        "the curve shows the fitted theoretical distribution."  # Execute this statement
    )

    # Column selection
    # Compute and store the result in col
    col = st.selectbox("Select column to analyze:", numeric_cols, key="dist_col")  # Column dropdown
    render_inline_unit_input(col, "dist")  # Ask user for unit — saved to session_state for axis labels
    series = drop_missing(df[col])  # Remove NaN values before analysis

    # Check condition and branch accordingly
    if len(series) < 10:  # Check condition
        # Show a red error message to the user
        st.error("Need at least 10 data points for distribution analysis.")  # Show a red error message
        # Exit the function early — stop execution here
        return  # Exit function early

    # Distribution selection
    # Compute and store the result in dist_type
    dist_type = st.selectbox(  # Store result in dist_type
        # Execute this operation
        "Select distribution:",  # Execute this statement
        # Execute this operation
        ["Normal", "Poisson", "Exponential", "Binomial", "Bernoulli", "Uniform"],  # Execute this statement
        # Compute and store the result in key
        key="dist_type_select"  # Store result in key
    )

    # Render formatted markdown text in the Streamlit UI
    st.markdown("---")  # Render formatted markdown text in the Streamlit UI

    # Check condition and branch accordingly
    if dist_type == "Normal":  # Check condition
        # Pandas data operation
        _render_normal(series, col)  # Execute this statement
    # Alternative condition check
    elif dist_type == "Poisson":  # Check alternative condition
        # Pandas data operation
        _render_poisson(series, col)  # Execute this statement
    # Alternative condition check
    elif dist_type == "Exponential":  # Check alternative condition
        # Pandas data operation
        _render_exponential(series, col)  # Execute this statement
    # Alternative condition check
    elif dist_type == "Binomial":  # Check alternative condition
        # Pandas data operation
        _render_binomial(series, col, df)  # Execute this statement
    # Alternative condition check
    elif dist_type == "Bernoulli":  # Check alternative condition
        # Pandas data operation
        _render_bernoulli(series, col, df)  # Execute this statement
    # Alternative condition check
    elif dist_type == "Uniform":  # Check alternative condition
        # Pandas data operation
        _render_uniform(series, col)  # Execute this statement


# =====================================================================
# Distribution renderers — all use real data
# =====================================================================

# Define the _render_normal function
def _render_normal(series: pd.Series, col: str):  # Define the _render_normal function
    """
    # Execute this operation
    Normal distribution fitted to real data using MLE.  # Execute this statement
    # Compute and store the result in MLE for Normal: μ
    MLE for Normal: μ = sample mean, σ = sample std dev  # Store result in MLE for Normal: μ
    """
    # Render formatted markdown text in the Streamlit UI
    st.markdown(f"### Normal Distribution fitted to '{col}'")  # Render formatted markdown text in the Streamlit UI

    # MLE parameter estimation: for Normal, MLE gives μ=mean, σ=std
    # Estimate Normal distribution parameters (μ, σ) from the data using MLE
    mu, sigma = sp.norm.fit(series)  # Fit Normal distribution parameters via MLE

    # Render formatted markdown text in the Streamlit UI
    st.markdown(f"""  # Render formatted markdown text in the Streamlit UI
    # Execute this operation
    **Parameters estimated from your data (MLE):**  # Execute this statement
    # Compute and store the result in - μ (mean)
    - μ (mean) = `{mu:.4f}` ← computed as: mean of '{col}'  # Store result in - μ (mean)
    # Compute and store the result in - σ (std dev)
    - σ (std dev) = `{sigma:.4f}` ← computed as: std dev of '{col}'  # Store result in - σ (std dev)
    """)

    # Create an evenly-spaced array of float values between two bounds
    x = np.linspace(series.min(), series.max(), 500)  # Create float array with even spacing
    # Compute Normal probability density function values
    pdf = sp.norm.pdf(x, loc=mu, scale=sigma)  # Compute Normal PDF values
    # Compute Normal cumulative distribution function values
    cdf = sp.norm.cdf(x, loc=mu, scale=sigma)  # Compute Normal CDF values

    # Create a new figure with one or more subplot axes
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # Create figure and subplot axes

    # PDF: real histogram + fitted curve
    # Draw a histogram showing value frequency distribution
    axes[0].hist(series, bins=30, density=True, alpha=0.5,  # Draw histogram of the data
                 # Compute and store the result in color
                 color="#4C72B0", edgecolor="white", label=f"Real data: {col}")  # Store result in color
    # Draw a line on the plot axes
    axes[0].plot(x, pdf, color="#C44E52", linewidth=2.5, label="Fitted Normal PDF")  # Draw a line on the axes
    # Set the title text for this subplot
    axes[0].set_title(f"PDF — Real data vs Normal fit")  # Set the subplot title
    # Label the x-axis
    axes[0].set_xlabel(label_with_unit(col))  # Label the x-axis
    # Label the y-axis
    axes[0].set_ylabel(density_label())  # Label the y-axis
    # Add a legend showing what each line/bar represents
    axes[0].legend(fontsize=8)  # Add legend to identify data series
    # Add faint grid lines to aid readability
    axes[0].grid(alpha=0.3)  # Add faint grid lines for readability

    # CDF: empirical vs fitted
    # Sort the array values in ascending order
    sorted_data = np.sort(series)  # Sort array values in ascending order
    # Create an evenly-spaced array of integer values
    empirical_cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)  # Create integer array with even spacing
    # Matplotlib or Plotly figure/axis operation
    axes[1].step(sorted_data, empirical_cdf, color="#4C72B0", linewidth=1.5, label="Empirical CDF")  # Draw step function line (for CDF plots)
    # Draw a line on the plot axes
    axes[1].plot(x, cdf, color="#C44E52", linewidth=2, linestyle="--", label="Fitted Normal CDF")  # Draw a line on the axes
    # Set the title text for this subplot
    axes[1].set_title("CDF — Empirical vs Fitted")  # Set the subplot title
    # Label the x-axis
    axes[1].set_xlabel(label_with_unit(col))  # Label the x-axis
    # Label the y-axis
    axes[1].set_ylabel(cdf_label())  # Label the y-axis
    # Add a legend showing what each line/bar represents
    axes[1].legend(fontsize=8)  # Add legend to identify data series
    # Add faint grid lines to aid readability
    axes[1].grid(alpha=0.3)  # Add faint grid lines for readability

    # Q-Q plot to check normality
    # Compute Q-Q plot data to visually assess normality
    (osm, osr), (slope, intercept, r) = sp.probplot(series, dist="norm")  # Compute Q-Q plot data for normality check
    # Draw scatter plot points
    axes[2].scatter(osm, osr, color="#4C72B0", s=10, alpha=0.6, label="Data quantiles")  # Draw scatter plot points
    # Draw a line on the plot axes
    axes[2].plot(osm, slope*np.array(osm)+intercept, color="#C44E52", linewidth=2, label="Perfect normal line")  # Draw a line on the axes
    # Set the title text for this subplot
    axes[2].set_title("Q-Q Plot (normality check)")  # Set the subplot title
    # Label the x-axis
    axes[2].set_xlabel("Theoretical quantiles")  # Label the x-axis
    # Label the y-axis
    axes[2].set_ylabel(f"Sample quantiles of {label_with_unit(col)}")  # Label the y-axis
    # Add a legend showing what each line/bar represents
    axes[2].legend(fontsize=8)  # Add legend to identify data series
    # Add faint grid lines to aid readability
    axes[2].grid(alpha=0.3)  # Add faint grid lines for readability

    # Adjust subplot spacing to prevent label overlap
    plt.tight_layout()  # Prevent subplot labels from overlapping
    # Render the matplotlib figure inside the Streamlit app
    st.pyplot(fig)  # Render the matplotlib figure in Streamlit
    # Close the figure to free memory after rendering
    plt.close(fig)  # Close figure to free memory

    # Goodness-of-fit
    # Run Kolmogorov-Smirnov test to check goodness-of-fit to a distribution
    ks_stat, ks_p = sp.kstest(series, "norm", args=(mu, sigma))  # Kolmogorov-Smirnov goodness-of-fit test

    # Execute this operation
    _show_dist_stats("Normal", {  # Execute this statement
        # Pandas data operation
        "n (sample size)": len(series),  # Execute this statement
        # Execute this operation
        "μ — estimated mean": round(mu, 4),  # Execute this statement
        # Execute this operation
        "σ — estimated std dev": round(sigma, 4),  # Execute this statement
        # Execute this operation
        "σ² — variance": round(sigma**2, 4),  # Execute this statement
        # Compute skewness — measure of distribution asymmetry
        "Skewness (data)": round(float(sp.skew(series)), 4),  # Compute skewness of the distribution
        # Compute kurtosis — measure of distribution tail heaviness
        "Kurtosis excess (data)": round(float(sp.kurtosis(series)), 4),  # Compute kurtosis of the distribution
        # Execute this operation
        "KS statistic": round(ks_stat, 4),  # Execute this statement
        # Execute this operation
        "KS p-value": round(ks_p, 4),  # Execute this statement
    # Execute this operation
    })  # Execute this statement

    # Open a context manager (auto-closes when done)
    with st.expander("📖 Formula & Interpretation"):  # Open context manager
        # Render a mathematical formula using LaTeX notation
        st.latex(r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}")  # Render a LaTeX mathematical formula
        # Render a mathematical formula using LaTeX notation
        st.latex(r"\hat{\mu}_{MLE} = \bar{x} = \frac{\sum x_i}{n}")  # Render a LaTeX mathematical formula
        # Render a mathematical formula using LaTeX notation
        st.latex(r"\hat{\sigma}_{MLE} = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n}}")  # Render a LaTeX mathematical formula
        # Check condition and branch accordingly
        if ks_p > 0.05:  # Check condition
            # Show a green success message to the user
            st.success(f"✅ KS test p={ks_p:.4f} > 0.05 → data is consistent with Normal distribution.")  # Show a green success message
        else:
            # Show a yellow warning message to the user
            st.warning(f"⚠️ KS test p={ks_p:.4f} ≤ 0.05 → data does NOT fit Normal distribution well.")  # Show a yellow warning message
        # Display text or data in the Streamlit UI
        st.write("**Q-Q Plot**: if points fall on the red line → data is approximately normal.")  # Display text or data in the Streamlit UI


# Define the _render_poisson function
def _render_poisson(series: pd.Series, col: str):  # Define the _render_poisson function
    """
    # Execute this operation
    Poisson distribution requires non-negative integer count data.  # Execute this statement
    # Execute this operation
    Checks compatibility first, explains if not suitable.  # Execute this statement
    # Compute and store the result in MLE for Poisson: λ
    MLE for Poisson: λ = sample mean  # Store result in MLE for Poisson: λ
    """
    # Render formatted markdown text in the Streamlit UI
    st.markdown(f"### Poisson Distribution fitted to '{col}'")  # Render formatted markdown text in the Streamlit UI

    # Check compatibility: Poisson requires non-negative values
    # Check condition and branch accordingly
    if series.min() < 0:  # Check condition
        # Show a red error message to the user
        st.error(f"""  # Show a red error message
        # Execute this operation
        ❌ **Poisson distribution cannot be applied to '{col}'.**  # Execute this statement

        # Execute this operation
        **Reason:** Poisson models count data — it requires all values to be ≥ 0.  # Execute this statement
        # Find the smallest value in the column
        Column '{col}' contains negative values (min = {series.min():.2f}).  # Store result in Column '{col}' contains negative values (min

        # Execute this operation
        **What to do:** Use Poisson for columns like: number of events, frequency counts, etc.  # Execute this statement
        # Execute this operation
        For '{col}', consider Normal or Exponential distribution instead.  # Execute this statement
        """)
        # Exit the function early — stop execution here
        return  # Exit function early

    # Warn if data looks non-integer (continuous data)
    # Round numeric values to the specified number of decimal places
    non_integer_pct = (series != series.round()).mean() * 100  # Store result in non_integer_pct
    # Check condition and branch accordingly
    if non_integer_pct > 10:  # Check condition
        # Show a yellow warning message to the user
        st.warning(f"""  # Show a yellow warning message
        # Execute this operation
        ⚠️ **Caution:** {non_integer_pct:.1f}% of values in '{col}' are not integers.  # Execute this statement

        # Execute this operation
        Poisson is designed for **discrete count data** (0, 1, 2, 3, ...).  # Execute this statement
        # Execute this operation
        For continuous data like '{col}', the fit may be poor.  # Execute this statement
        # Execute this operation
        Proceeding anyway — check the KS p-value for fit quality.  # Execute this statement
        """)

    # MLE for Poisson: λ = sample mean
    # Compute the arithmetic mean of the column or series
    lam = series.mean()  # Store result in lam
    # Render formatted markdown text in the Streamlit UI
    st.markdown(f"""  # Render formatted markdown text in the Streamlit UI
    # Execute this operation
    **Parameter estimated from your data (MLE):**  # Execute this statement
    # Compute the arithmetic mean of the column or series
    - λ = `{lam:.4f}` ← computed as: mean of '{col}' = {series.mean():.4f}  # Store result in - λ
    # Compute and store the result in - For Poisson: MLE gives λ̂
    - For Poisson: MLE gives λ̂ = x̄ (sample mean)  # Store result in - For Poisson: MLE gives λ̂
    """)
    # Render a mathematical formula using LaTeX notation
    st.latex(r"\hat{\lambda}_{MLE} = \bar{x} = \frac{\sum x_i}{n}")  # Render a LaTeX mathematical formula

    # Create an evenly-spaced array of integer values
    x_int = np.arange(0, int(series.max()) + 2)  # Create integer array with even spacing
    # Compute Poisson probability mass function values
    pmf = sp.poisson.pmf(x_int, mu=lam)  # Compute Poisson PMF values
    # Compute Poisson cumulative distribution function values
    cdf_vals = sp.poisson.cdf(x_int, mu=lam)  # Compute Poisson CDF values

    # Create a new figure with one or more subplot axes
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))  # Create figure and subplot axes

    # Empirical frequency vs fitted PMF
    # Count how many times each unique value appears
    val_counts = series.round().value_counts(normalize=True).sort_index()  # Store result in val_counts
    # Draw a bar chart
    axes[0].bar(val_counts.index, val_counts.values, alpha=0.5,  # Draw bar chart
                # Compute and store the result in color
                color="#4C72B0", label=f"Real data: {col}", width=0.4)  # Store result in color
    # Draw a line on the plot axes
    axes[0].plot(x_int, pmf, "ro-", markersize=5, linewidth=1.5,  # Draw a line on the axes
                 # Compute and store the result in label
                 label=f"Fitted Poisson PMF (λ={lam:.2f})")  # Store result in label
    # Set the title text for this subplot
    axes[0].set_title(f"PMF — Real data vs Poisson fit")  # Set the subplot title
    # Label the x-axis
    axes[0].set_xlabel(f"{label_with_unit(col)}  (k = number of events)")  # Label the x-axis
    # Label the y-axis
    axes[0].set_ylabel(probability_label())  # Label the y-axis
    # Add a legend showing what each line/bar represents
    axes[0].legend(fontsize=8)  # Add legend to identify data series
    # Add faint grid lines to aid readability
    axes[0].grid(axis="y", alpha=0.3)  # Add faint grid lines for readability

    # Empirical CDF vs fitted CDF
    # Sort the array values in ascending order
    sorted_data = np.sort(series)  # Sort array values in ascending order
    # Create an evenly-spaced array of integer values
    empirical_cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)  # Create integer array with even spacing
    # Matplotlib or Plotly figure/axis operation
    axes[1].step(sorted_data, empirical_cdf, color="#4C72B0", linewidth=1.5, label="Empirical CDF")  # Draw step function line (for CDF plots)
    # Matplotlib or Plotly figure/axis operation
    axes[1].step(x_int, cdf_vals, color="#C44E52", linewidth=2, linestyle="--",  # Draw step function line (for CDF plots)
                 # Compute and store the result in label
                 label=f"Fitted Poisson CDF")  # Store result in label
    # Set the title text for this subplot
    axes[1].set_title("CDF — Empirical vs Fitted")  # Set the subplot title
    # Label the x-axis
    axes[1].set_xlabel(label_with_unit(col))  # Label the x-axis
    # Label the y-axis
    axes[1].set_ylabel(cdf_label())  # Label the y-axis
    # Add a legend showing what each line/bar represents
    axes[1].legend(fontsize=8)  # Add legend to identify data series
    # Add faint grid lines to aid readability
    axes[1].grid(alpha=0.3)  # Add faint grid lines for readability

    # Adjust subplot spacing to prevent label overlap
    plt.tight_layout()  # Prevent subplot labels from overlapping
    # Render the matplotlib figure inside the Streamlit app
    st.pyplot(fig)  # Render the matplotlib figure in Streamlit
    # Close the figure to free memory after rendering
    plt.close(fig)  # Close figure to free memory

    # Compute Poisson cumulative distribution function values
    ks_stat, ks_p = sp.kstest(series, lambda x: sp.poisson.cdf(x, mu=lam))  # Compute Poisson CDF values

    # Execute this operation
    _show_dist_stats("Poisson", {  # Execute this statement
        # Pandas data operation
        "n (sample size)": len(series),  # Execute this statement
        # Compute and store the result in "λ — estimated rate (
        "λ — estimated rate (= mean)": round(lam, 4),  # Store result in "λ — estimated rate (
        # Compute the arithmetic mean of the column or series
        "Data mean": round(float(series.mean()), 4),  # Execute this statement
        # Compute the variance of the column
        "Data variance (should ≈ λ)": round(float(series.var()), 4),  # Execute this statement
        # Compute the square root
        "Skewness (theoretical: 1/√λ)": round(1/np.sqrt(lam), 4),  # Compute square root
        # Compute skewness — measure of distribution asymmetry
        "Skewness (data)": round(float(sp.skew(series)), 4),  # Compute skewness of the distribution
        # Execute this operation
        "KS statistic": round(ks_stat, 4),  # Execute this statement
        # Execute this operation
        "KS p-value": round(ks_p, 4),  # Execute this statement
    # Execute this operation
    })  # Execute this statement

    # Open a context manager (auto-closes when done)
    with st.expander("📖 Formula & Interpretation"):  # Open context manager
        # Render a mathematical formula using LaTeX notation
        st.latex(r"P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}")  # Render a LaTeX mathematical formula
        # Check condition and branch accordingly
        if ks_p > 0.05:  # Check condition
            # Show a green success message to the user
            st.success(f"✅ KS p={ks_p:.4f} > 0.05 → data is consistent with Poisson.")  # Show a green success message
        else:
            # Show a yellow warning message to the user
            st.warning(f"⚠️ KS p={ks_p:.4f} ≤ 0.05 → poor fit. Check if data variance ≈ mean.")  # Show a yellow warning message
        # Display text or data in the Streamlit UI
        st.write("**Key check for Poisson:** mean ≈ variance. If they differ a lot, Poisson is not the right model.")  # Display text or data in the Streamlit UI


# Define the _render_exponential function
def _render_exponential(series: pd.Series, col: str):  # Define the _render_exponential function
    """
    # Execute this operation
    Exponential distribution requires non-negative data.  # Execute this statement
    # Compute and store the result in MLE for Exponential: λ
    MLE for Exponential: λ = 1 / sample mean  # Store result in MLE for Exponential: λ
    """
    # Render formatted markdown text in the Streamlit UI
    st.markdown(f"### Exponential Distribution fitted to '{col}'")  # Render formatted markdown text in the Streamlit UI

    # Check condition and branch accordingly
    if series.min() < 0:  # Check condition
        # Show a red error message to the user
        st.error(f"""  # Show a red error message
        # Execute this operation
        ❌ **Exponential distribution cannot be applied to '{col}'.**  # Execute this statement

        # Execute this operation
        **Reason:** Exponential models time/duration between events — requires all values ≥ 0.  # Execute this statement
        # Find the smallest value in the column
        Column '{col}' has negative values (min = {series.min():.2f}).  # Store result in Column '{col}' has negative values (min

        # Execute this operation
        **What to do:** Use Exponential for columns representing durations, waiting times, or gaps.  # Execute this statement
        # Execute this operation
        For '{col}', try Normal distribution instead.  # Execute this statement
        """)
        # Exit the function early — stop execution here
        return  # Exit function early

    # MLE for Exponential: loc=0, scale=1/λ=mean
    loc, scale = sp.expon.fit(series, floc=0)   # floc=0 forces location=0 (standard exponential)
    # Compute and store the result in lam
    lam = 1 / scale  # Store result in lam

    # Render formatted markdown text in the Streamlit UI
    st.markdown(f"""  # Render formatted markdown text in the Streamlit UI
    # Execute this operation
    **Parameters estimated from your data (MLE):**  # Execute this statement
    # Compute the arithmetic mean of the column or series
    - λ (rate) = `{lam:.6f}` ← computed as: 1 / mean = 1 / {series.mean():.4f}  # Store result in - λ (rate)
    # Compute and store the result in - Mean
    - Mean = 1/λ = `{scale:.4f}`  # Store result in - Mean
    """)
    # Render a mathematical formula using LaTeX notation
    st.latex(r"\hat{\lambda}_{MLE} = \frac{1}{\bar{x}}")  # Render a LaTeX mathematical formula

    # Create an evenly-spaced array of float values between two bounds
    x = np.linspace(0, series.max(), 500)  # Create float array with even spacing
    # Compute Exponential probability density function values
    pdf = sp.expon.pdf(x, loc=loc, scale=scale)  # Compute Exponential PDF values
    # Compute Exponential CDF values
    cdf_vals = sp.expon.cdf(x, loc=loc, scale=scale)  # Compute Exponential CDF values

    # Create a new figure with one or more subplot axes
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))  # Create figure and subplot axes

    # Draw a histogram showing value frequency distribution
    axes[0].hist(series, bins=30, density=True, alpha=0.5,  # Draw histogram of the data
                 # Compute and store the result in color
                 color="#55A868", edgecolor="white", label=f"Real data: {col}")  # Store result in color
    # Draw a line on the plot axes
    axes[0].plot(x, pdf, color="#C44E52", linewidth=2.5,  # Draw a line on the axes
                 # Compute and store the result in label
                 label=f"Fitted Exp PDF (λ={lam:.4f})")  # Store result in label
    # Set the title text for this subplot
    axes[0].set_title("PDF — Real data vs Exponential fit")  # Set the subplot title
    # Label the x-axis
    axes[0].set_xlabel(label_with_unit(col))  # Label the x-axis
    # Label the y-axis
    axes[0].set_ylabel(density_label())  # Label the y-axis
    # Add a legend showing what each line/bar represents
    axes[0].legend(fontsize=8)  # Add legend to identify data series
    # Add faint grid lines to aid readability
    axes[0].grid(alpha=0.3)  # Add faint grid lines for readability

    # Sort the array values in ascending order
    sorted_data = np.sort(series)  # Sort array values in ascending order
    # Create an evenly-spaced array of integer values
    emp_cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)  # Create integer array with even spacing
    # Matplotlib or Plotly figure/axis operation
    axes[1].step(sorted_data, emp_cdf, color="#55A868", linewidth=1.5, label="Empirical CDF")  # Draw step function line (for CDF plots)
    # Draw a line on the plot axes
    axes[1].plot(x, cdf_vals, color="#C44E52", linewidth=2, linestyle="--",  # Draw a line on the axes
                 # Compute and store the result in label
                 label="Fitted Exponential CDF")  # Store result in label
    # Set the title text for this subplot
    axes[1].set_title("CDF — Empirical vs Fitted")  # Set the subplot title
    # Label the x-axis
    axes[1].set_xlabel(label_with_unit(col))  # Label the x-axis
    # Label the y-axis
    axes[1].set_ylabel(cdf_label())  # Label the y-axis
    # Add a legend showing what each line/bar represents
    axes[1].legend(fontsize=8)  # Add legend to identify data series
    # Add faint grid lines to aid readability
    axes[1].grid(alpha=0.3)  # Add faint grid lines for readability

    # Adjust subplot spacing to prevent label overlap
    plt.tight_layout()  # Prevent subplot labels from overlapping
    # Render the matplotlib figure inside the Streamlit app
    st.pyplot(fig)  # Render the matplotlib figure in Streamlit
    # Close the figure to free memory after rendering
    plt.close(fig)  # Close figure to free memory

    # Run Kolmogorov-Smirnov test to check goodness-of-fit to a distribution
    ks_stat, ks_p = sp.kstest(series, "expon", args=(loc, scale))  # Kolmogorov-Smirnov goodness-of-fit test

    # Execute this operation
    _show_dist_stats("Exponential", {  # Execute this statement
        # Pandas data operation
        "n (sample size)": len(series),  # Execute this statement
        # Execute this operation
        "λ — estimated rate": round(lam, 6),  # Execute this statement
        # Execute this operation
        "Mean (1/λ)": round(scale, 4),  # Execute this statement
        # Execute this operation
        "Variance (1/λ²)": round(scale**2, 4),  # Execute this statement
        # Execute this operation
        "Skewness (theoretical: 2)": 2,  # Execute this statement
        # Compute skewness — measure of distribution asymmetry
        "Skewness (data)": round(float(sp.skew(series)), 4),  # Compute skewness of the distribution
        # Execute this operation
        "Kurtosis excess (theoretical: 6)": 6,  # Execute this statement
        # Compute kurtosis — measure of distribution tail heaviness
        "Kurtosis excess (data)": round(float(sp.kurtosis(series)), 4),  # Compute kurtosis of the distribution
        # Execute this operation
        "KS statistic": round(ks_stat, 4),  # Execute this statement
        # Execute this operation
        "KS p-value": round(ks_p, 4),  # Execute this statement
    # Execute this operation
    })  # Execute this statement

    # Open a context manager (auto-closes when done)
    with st.expander("📖 Formula & Interpretation"):  # Open context manager
        # Render a mathematical formula using LaTeX notation
        st.latex(r"f(x) = \lambda e^{-\lambda x}, \quad x \geq 0")  # Render a LaTeX mathematical formula
        # Render a mathematical formula using LaTeX notation
        st.latex(r"\hat{\lambda}_{MLE} = \frac{1}{\bar{x}} = \frac{n}{\sum x_i}")  # Render a LaTeX mathematical formula
        # Check condition and branch accordingly
        if ks_p > 0.05:  # Check condition
            # Show a green success message to the user
            st.success(f"✅ KS p={ks_p:.4f} > 0.05 → data is consistent with Exponential.")  # Show a green success message
        else:
            # Show a yellow warning message to the user
            st.warning(f"⚠️ KS p={ks_p:.4f} ≤ 0.05 → poor fit. Exponential expects skewness≈2.")  # Show a yellow warning message


# Define the _render_binomial function
def _render_binomial(series: pd.Series, col: str, df: pd.DataFrame):  # Define the _render_binomial function
    """
    # Execute this operation
    Binomial requires knowing n (number of trials).  # Execute this statement
    # Execute this operation
    Estimates p from data. Checks compatibility.  # Execute this statement
    """
    # Render formatted markdown text in the Streamlit UI
    st.markdown(f"### Binomial Distribution fitted to '{col}'")  # Render formatted markdown text in the Streamlit UI

    # Check: values must be non-negative integers
    # Check condition and branch accordingly
    if series.min() < 0:  # Check condition
        # Show a red error message to the user
        st.error(f"""  # Show a red error message
        # Execute this operation
        ❌ **Binomial distribution cannot be applied to '{col}'.**  # Execute this statement

        # Execute this operation
        **Reason:** Binomial counts successes in n trials — requires values ≥ 0.  # Execute this statement
        # Find the smallest value in the column
        Column '{col}' has negative values (min = {series.min():.2f}).  # Store result in Column '{col}' has negative values (min
        """)
        # Exit the function early — stop execution here
        return  # Exit function early

    # Round numeric values to the specified number of decimal places
    non_int_pct = (series != series.round()).mean() * 100  # Store result in non_int_pct
    # Check condition and branch accordingly
    if non_int_pct > 5:  # Check condition
        # Show a yellow warning message to the user
        st.warning(f"""  # Show a yellow warning message
        # Execute this operation
        ⚠️ **{non_int_pct:.1f}% of values in '{col}' are not integers.**  # Execute this statement

        # Execute this operation
        Binomial counts discrete successes (0, 1, 2, ..., n).  # Execute this statement
        # Execute this operation
        Column '{col}' appears to be continuous data.  # Execute this statement
        # Execute this operation
        Proceeding with rounded values — fit quality may be poor.  # Execute this statement
        """)
        # Convert column values to a specified data type
        series = series.round().astype(int)  # Store result in series

    # n must be specified by user — it's the number of trials per observation
    # Find the largest value in the column
    n_max = int(series.max())  # Store result in n_max
    # Show an informational blue message box
    st.info(f"""  # Show an informational blue message box
    # Execute this operation
    ℹ️ **Binomial requires knowing n (number of trials per observation).**  # Execute this statement

    # Execute this operation
    The max value in '{col}' is {n_max} — this is used as the upper bound.  # Execute this statement
    # Execute this operation
    You can adjust n below if you know the actual number of trials.  # Execute this statement
    """)
    # Compute and store the result in n
    n = st.slider("n (number of trials per observation):", n_max, n_max*3, n_max, 1, key="binom_n_real")  # Store result in n

    # MLE for Binomial: p̂ = mean / n
    # Compute the arithmetic mean of the column or series
    p_hat = series.mean() / n  # Store result in p_hat
    p_hat = np.clip(p_hat, 0.001, 0.999)   # Clip to valid range

    # Render formatted markdown text in the Streamlit UI
    st.markdown(f"""  # Render formatted markdown text in the Streamlit UI
    # Execute this operation
    **Parameters estimated from your data (MLE):**  # Execute this statement
    # Compute and store the result in - n
    - n = `{n}` (number of trials — set above)  # Store result in - n
    # Compute the arithmetic mean of the column or series
    - p̂ = mean / n = `{series.mean():.4f}` / `{n}` = `{p_hat:.4f}`  # Store result in - p̂
    """)
    # Render a mathematical formula using LaTeX notation
    st.latex(r"\hat{p}_{MLE} = \frac{\bar{x}}{n}")  # Render a LaTeX mathematical formula

    # Create an evenly-spaced array of integer values
    x_vals = np.arange(0, n + 1)  # Create integer array with even spacing
    # Compute Binomial probability mass function values
    pmf = sp.binom.pmf(x_vals, n=n, p=p_hat)  # Compute Binomial PMF values
    # Compute Binomial cumulative distribution function values
    cdf_vals = sp.binom.cdf(x_vals, n=n, p=p_hat)  # Compute Binomial CDF values

    # Create a new figure with one or more subplot axes
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))  # Create figure and subplot axes

    # Count how many times each unique value appears
    val_counts = series.value_counts(normalize=True).sort_index()  # Store result in val_counts
    # Draw a bar chart
    axes[0].bar(val_counts.index, val_counts.values, alpha=0.5,  # Draw bar chart
                # Compute and store the result in color
                color="#4C72B0", label=f"Real data: {col}", width=0.4)  # Store result in color
    # Draw a line on the plot axes
    axes[0].plot(x_vals, pmf, "ro-", markersize=4, linewidth=1.5,  # Draw a line on the axes
                 # Compute and store the result in label
                 label=f"Fitted B(n={n}, p={p_hat:.3f})")  # Store result in label
    # Set the title text for this subplot
    axes[0].set_title("PMF — Real data vs Binomial fit")  # Set the subplot title
    # Label the x-axis
    axes[0].set_xlabel(f"{label_with_unit(col)}  (k = number of successes)")  # Label the x-axis
    # Label the y-axis
    axes[0].set_ylabel(probability_label())  # Label the y-axis
    # Add a legend showing what each line/bar represents
    axes[0].legend(fontsize=8)  # Add legend to identify data series
    # Add faint grid lines to aid readability
    axes[0].grid(axis="y", alpha=0.3)  # Add faint grid lines for readability

    # Sort the array values in ascending order
    sorted_data = np.sort(series)  # Sort array values in ascending order
    # Create an evenly-spaced array of integer values
    emp_cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)  # Create integer array with even spacing
    # Matplotlib or Plotly figure/axis operation
    axes[1].step(sorted_data, emp_cdf, color="#4C72B0", linewidth=1.5, label="Empirical CDF")  # Draw step function line (for CDF plots)
    # Matplotlib or Plotly figure/axis operation
    axes[1].step(x_vals, cdf_vals, color="#C44E52", linewidth=2, linestyle="--",  # Draw step function line (for CDF plots)
                 # Compute and store the result in label
                 label="Fitted Binomial CDF")  # Store result in label
    # Set the title text for this subplot
    axes[1].set_title("CDF — Empirical vs Fitted")  # Set the subplot title
    # Label the x-axis
    axes[1].set_xlabel(label_with_unit(col))  # Label the x-axis
    # Label the y-axis
    axes[1].set_ylabel(cdf_label())  # Label the y-axis
    # Add a legend showing what each line/bar represents
    axes[1].legend(fontsize=8)  # Add legend to identify data series
    # Add faint grid lines to aid readability
    axes[1].grid(alpha=0.3)  # Add faint grid lines for readability

    # Adjust subplot spacing to prevent label overlap
    plt.tight_layout()  # Prevent subplot labels from overlapping
    # Render the matplotlib figure inside the Streamlit app
    st.pyplot(fig)  # Render the matplotlib figure in Streamlit
    # Close the figure to free memory after rendering
    plt.close(fig)  # Close figure to free memory

    # Execute this operation
    _show_dist_stats("Binomial", {  # Execute this statement
        # Execute this operation
        "n (trials)": n,  # Execute this statement
        # Execute this operation
        "p̂ — estimated probability": round(p_hat, 4),  # Execute this statement
        # Pandas data operation
        "n (sample size)": len(series),  # Execute this statement
        # Execute this operation
        "Mean (np)": round(n * p_hat, 4),  # Execute this statement
        # Execute this operation
        "Variance (np(1-p))": round(n * p_hat * (1 - p_hat), 4),  # Execute this statement
        # Compute skewness — measure of distribution asymmetry
        "Skewness (data)": round(float(sp.skew(series)), 4),  # Compute skewness of the distribution
        # Compute kurtosis — measure of distribution tail heaviness
        "Kurtosis excess (data)": round(float(sp.kurtosis(series)), 4),  # Compute kurtosis of the distribution
    # Execute this operation
    })  # Execute this statement

    # Open a context manager (auto-closes when done)
    with st.expander("📖 Formula & Interpretation"):  # Open context manager
        # Render a mathematical formula using LaTeX notation
        st.latex(r"P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}")  # Render a LaTeX mathematical formula
        # Render a mathematical formula using LaTeX notation
        st.latex(r"\hat{p}_{MLE} = \frac{\bar{x}}{n}")  # Render a LaTeX mathematical formula


# Define the _render_bernoulli function
def _render_bernoulli(series: pd.Series, col: str, df: pd.DataFrame):  # Define the _render_bernoulli function
    """
    # Execute this operation
    Bernoulli requires binary data (only 0 and 1).  # Execute this statement
    # Execute this operation
    Checks if column is binary. If not, suggests a categorical column or explains why.  # Execute this statement
    """
    # Render formatted markdown text in the Streamlit UI
    st.markdown(f"### Bernoulli Distribution fitted to '{col}'")  # Render formatted markdown text in the Streamlit UI

    # Remove rows with missing (NaN) values
    unique_vals = series.dropna().unique()  # Remove rows with NaN values

    # Check if column is binary (only 0 and 1)
    # Compute and store the result in is_binary_numeric
    is_binary_numeric = set(unique_vals).issubset({0, 1, 0.0, 1.0})  # Store result in is_binary_numeric

    # Check condition and branch accordingly
    if not is_binary_numeric:  # Check condition
        # Show a red error message to the user
        st.error(f"""  # Show a red error message
        # Execute this operation
        ❌ **Bernoulli distribution cannot be directly applied to '{col}'.**  # Execute this statement

        # Execute this operation
        **Reason:** Bernoulli models a single trial with exactly two outcomes: 0 (failure) or 1 (success).  # Execute this statement
        # Execute this operation
        Column '{col}' has {len(unique_vals)} unique values: {sorted(unique_vals[:5])}{' ...' if len(unique_vals)>5 else ''} — not binary.  # Execute this statement

        # Execute this operation
        **What you can do instead:**  # Execute this statement
        """)

        # Suggest categorical columns as alternatives
        # Pandas data operation
        cat_cols = get_categorical_columns(df)  # Store result in cat_cols
        # Check condition and branch accordingly
        if cat_cols:  # Check condition
            # Show an informational blue message box
            st.info(f"""  # Show an informational blue message box
            # Execute this operation
            👉 **Option A — Use a categorical column:**  # Execute this statement
            # Execute this operation
            Your dataset has these categorical columns: {cat_cols}  # Execute this statement
            # Execute this operation
            Select one of them below to convert to 0/1 and apply Bernoulli.  # Execute this statement
            """)
            # Compute and store the result in cat_col
            cat_col = st.selectbox("Select categorical column:", cat_cols, key="bern_cat_alt")  # Store result in cat_col
            # Remove rows with missing (NaN) values
            cat_series = df[cat_col].dropna()  # Remove rows with NaN values
            # Get all unique values in the column as an array
            cat_unique = cat_series.unique().tolist()  # Store result in cat_unique
            # Check condition and branch accordingly
            if len(cat_unique) == 2:  # Check condition
                # Compute and store the result in success_val
                success_val = st.selectbox(f"Which value counts as SUCCESS (1)?", cat_unique, key="bern_success")  # Store result in success_val
                # Convert column values to a specified data type
                binary = (cat_series == success_val).astype(int)  # Store result in binary
                # Show a green success message to the user
                st.success(f"✅ Converted '{cat_col}': '{success_val}'=1, other=0. Now applying Bernoulli.")  # Show a green success message
                # Compute and store the result in _apply_bernoulli(binary, f"{cat_col} (binary: {success_val}
                _apply_bernoulli(binary, f"{cat_col} (binary: {success_val}=1)")  # Store result in _apply_bernoulli(binary, f"{cat_col} (binary: {success_val}
            else:
                # Show a yellow warning message to the user
                st.warning(f"Column '{cat_col}' has {len(cat_unique)} values — need exactly 2 for Bernoulli.")  # Show a yellow warning message

        # Show an informational blue message box
        st.info("""  # Show an informational blue message box
        # Execute this operation
        👉 **Option B — Threshold numeric column:**  # Execute this statement
        # Compute and store the result in Convert '{col}' to binary by choosing a threshold (e.g. Salary > 55000
        Convert '{col}' to binary by choosing a threshold (e.g. Salary > 55000 = 1, else 0).  # Store result in Salary > 55000
        """)
        # Compute and store the result in threshold
        threshold = st.number_input(  # Store result in threshold
            # Execute this operation
            f"Threshold: '{col}' > threshold → 1, else → 0",  # Execute this statement
            # Find the median (middle) value of the column
            value=float(series.median()), key="bern_threshold"  # Store result in value
        )
        # Convert column values to a specified data type
        binary_threshold = (series > threshold).astype(int)  # Store result in binary_threshold
        # Compute and store the result in n_ones
        n_ones = binary_threshold.sum()  # Store result in n_ones
        # Compute and store the result in n_zeros
        n_zeros = len(binary_threshold) - n_ones  # Store result in n_zeros
        # Display text or data in the Streamlit UI
        st.write(f"Result: {n_ones} ones ({100*n_ones/len(binary_threshold):.1f}%), {n_zeros} zeros")  # Display text or data in the Streamlit UI
        # Check condition and branch accordingly
        if st.button("Apply Bernoulli with this threshold"):  # Check condition
            # Execute this operation
            _apply_bernoulli(binary_threshold, f"{col} > {threshold:.2f}")  # Execute this statement
        # Exit the function early — stop execution here
        return  # Exit function early

    # If already binary, apply directly
    # Convert column values to a specified data type
    _apply_bernoulli(series.astype(int), col)  # Execute this statement


# Define the _apply_bernoulli function
def _apply_bernoulli(binary: pd.Series, label: str):  # Define the _apply_bernoulli function
    """Apply Bernoulli distribution to a binary (0/1) series."""
    # MLE for Bernoulli: p̂ = proportion of successes
    # Compute and store the result in p_hat
    p_hat = binary.mean()  # Store result in p_hat

    # Render formatted markdown text in the Streamlit UI
    st.markdown(f"""  # Render formatted markdown text in the Streamlit UI
    # Execute this operation
    **Parameter estimated from your data (MLE):**  # Execute this statement
    # Compute and store the result in - p̂
    - p̂ = proportion of 1s = `{binary.sum()}` / `{len(binary)}` = `{p_hat:.4f}`  # Store result in - p̂
    """)
    # Render a mathematical formula using LaTeX notation
    st.latex(r"\hat{p}_{MLE} = \frac{\text{number of successes}}{n} = \bar{x}")  # Render a LaTeX mathematical formula

    # Convert a list or sequence into a numpy array
    x = np.array([0, 1])  # Convert list to numpy array
    # Compute Bernoulli probability mass function values
    pmf = sp.bernoulli.pmf(x, p=p_hat)  # Compute Bernoulli PMF values
    # Compute Bernoulli CDF values
    cdf_vals = sp.bernoulli.cdf(x, p=p_hat)  # Compute Bernoulli CDF values

    # Create a new figure with one or more subplot axes
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # Create figure and subplot axes

    # Real counts vs fitted PMF
    # Compute and store the result in real_counts
    real_counts = binary.value_counts(normalize=True).reindex([0, 1], fill_value=0)  # Store result in real_counts
    # Draw a bar chart
    axes[0].bar(["Failure (0)", "Success (1)"], real_counts.values,  # Draw bar chart
                # Compute and store the result in color
                color=["#C44E52", "#55A868"], edgecolor="white", alpha=0.6,  # Store result in color
                # Compute and store the result in label
                label=f"Real data: {label}")  # Store result in label
    # Draw a line on the plot axes
    axes[0].plot(["Failure (0)", "Success (1)"], pmf, "ko--", markersize=8,  # Draw a line on the axes
                 # Compute and store the result in linewidth
                 linewidth=1.5, label=f"Fitted Ber(p={p_hat:.3f})")  # Store result in linewidth
    # Set the title text for this subplot
    axes[0].set_title(f"PMF — Real data vs Bernoulli fit")  # Set the subplot title
    # Label the y-axis
    axes[0].set_ylabel(probability_label())  # Label the y-axis
    # Set the y-axis display range
    axes[0].set_ylim(0, 1.1)  # Set y-axis display range
    # Add a legend showing what each line/bar represents
    axes[0].legend(fontsize=8)  # Add legend to identify data series
    # Add faint grid lines to aid readability
    axes[0].grid(axis="y", alpha=0.3)  # Add faint grid lines for readability

    # Matplotlib or Plotly figure/axis operation
    axes[1].step([0, 1, 2], [0] + list(cdf_vals), where="post",  # Draw step function line (for CDF plots)
                 # Compute and store the result in color
                 color="#C44E52", linewidth=2, label="Fitted Bernoulli CDF")  # Store result in color
    # Set the title text for this subplot
    axes[1].set_title("CDF")  # Set the subplot title
    # Label the x-axis
    axes[1].set_xlabel(f"x  (0=failure, 1=success)")  # Label x-axis: binary outcome, no unit needed
    # Label the y-axis
    axes[1].set_ylabel(cdf_label())  # Label the y-axis
    # Specify exact tick positions on the x-axis
    axes[1].set_xticks([0, 1])  # Specify tick positions on x-axis
    # Add a legend showing what each line/bar represents
    axes[1].legend(fontsize=8)  # Add legend to identify data series
    # Add faint grid lines to aid readability
    axes[1].grid(alpha=0.3)  # Add faint grid lines for readability

    # Adjust subplot spacing to prevent label overlap
    plt.tight_layout()  # Prevent subplot labels from overlapping
    # Render the matplotlib figure inside the Streamlit app
    st.pyplot(fig)  # Render the matplotlib figure in Streamlit
    # Close the figure to free memory after rendering
    plt.close(fig)  # Close figure to free memory

    # Execute this operation
    _show_dist_stats("Bernoulli", {  # Execute this statement
        # Execute this operation
        "n (sample size)": len(binary),  # Execute this statement
        # Execute this operation
        "p̂ — estimated success prob": round(p_hat, 4),  # Execute this statement
        # Execute this operation
        "Count of 1s (successes)": int(binary.sum()),  # Execute this statement
        # Compute and store the result in "Count of 0s (failures)": int((binary
        "Count of 0s (failures)": int((binary == 0).sum()),  # Store result in "Count of 0s (failures)": int((binary
        # Compute and store the result in "Mean (
        "Mean (= p̂)": round(p_hat, 4),  # Store result in "Mean (
        # Execute this operation
        "Variance (p(1-p))": round(p_hat * (1 - p_hat), 4),  # Execute this statement
        # Compute the square root
        "Skewness (theoretical)": round((1-2*p_hat)/np.sqrt(p_hat*(1-p_hat)), 4) if 0 < p_hat < 1 else 0,  # Compute square root
    # Execute this operation
    })  # Execute this statement

    # Open a context manager (auto-closes when done)
    with st.expander("📖 Formula & Interpretation"):  # Open context manager
        # Render a mathematical formula using LaTeX notation
        st.latex(r"P(X = k) = p^k (1-p)^{1-k}, \quad k \in \{0, 1\}")  # Render a LaTeX mathematical formula
        # Render a mathematical formula using LaTeX notation
        st.latex(r"\hat{p}_{MLE} = \bar{x} = \frac{\sum x_i}{n}")  # Render a LaTeX mathematical formula


# Define the _render_uniform function
def _render_uniform(series: pd.Series, col: str):  # Define the _render_uniform function
    """
    # Execute this operation
    Uniform distribution: parameters estimated from data.  # Execute this statement
    # Compute and store the result in MLE for Uniform: a
    MLE for Uniform: a = min, b = max  # Store result in MLE for Uniform: a
    # Execute this operation
    Note: any continuous data can be compared to Uniform — we explain fit quality.  # Execute this statement
    """
    # Render formatted markdown text in the Streamlit UI
    st.markdown(f"### Uniform Distribution fitted to '{col}'")  # Render formatted markdown text in the Streamlit UI

    # MLE for Uniform: a=min, b=max
    # Find the smallest value in the column
    a = series.min()  # Store result in a
    # Find the largest value in the column
    b = series.max()  # Store result in b

    # Render formatted markdown text in the Streamlit UI
    st.markdown(f"""  # Render formatted markdown text in the Streamlit UI
    # Execute this operation
    **Parameters estimated from your data (MLE):**  # Execute this statement
    # Compute and store the result in - a (lower bound)
    - a (lower bound) = `{a:.4f}` ← min of '{col}'  # Store result in - a (lower bound)
    # Compute and store the result in - b (upper bound)
    - b (upper bound) = `{b:.4f}` ← max of '{col}'  # Store result in - b (upper bound)

    # Compute and store the result in **Note:** For Uniform distribution, MLE gives a
    **Note:** For Uniform distribution, MLE gives a = min(x), b = max(x).  # Store result in **Note:** For Uniform distribution, MLE gives a
    # Execute this operation
    A Uniform distribution assumes all values in [a, b] are equally likely.  # Execute this statement
    # Execute this operation
    If your data is concentrated in the middle, the fit will be poor (expected).  # Execute this statement
    """)
    # Render a mathematical formula using LaTeX notation
    st.latex(r"\hat{a}_{MLE} = \min(x_i), \quad \hat{b}_{MLE} = \max(x_i)")  # Render a LaTeX mathematical formula

    # Create an evenly-spaced array of float values between two bounds
    x = np.linspace(a - 0.05*(b-a), b + 0.05*(b-a), 500)  # Create float array with even spacing
    # Compute Uniform distribution probability density values
    pdf = sp.uniform.pdf(x, loc=a, scale=b-a)  # Compute Uniform PDF values
    # Compute Uniform CDF values
    cdf_vals = sp.uniform.cdf(x, loc=a, scale=b-a)  # Compute Uniform CDF values

    # Create a new figure with one or more subplot axes
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))  # Create figure and subplot axes

    # Draw a histogram showing value frequency distribution
    axes[0].hist(series, bins=30, density=True, alpha=0.5,  # Draw histogram of the data
                 # Compute and store the result in color
                 color="#CCB974", edgecolor="white", label=f"Real data: {col}")  # Store result in color
    # Draw a line on the plot axes
    axes[0].plot(x, pdf, color="#C44E52", linewidth=2.5,  # Draw a line on the axes
                 # Compute and store the result in label
                 label=f"Fitted Uniform PDF\n[{a:.2f}, {b:.2f}]")  # Store result in label
    # Set the title text for this subplot
    axes[0].set_title("PDF — Real data vs Uniform fit")  # Set the subplot title
    # Label the x-axis
    axes[0].set_xlabel(label_with_unit(col))  # Label the x-axis
    # Label the y-axis
    axes[0].set_ylabel(density_label())  # Label the y-axis
    # Add a legend showing what each line/bar represents
    axes[0].legend(fontsize=8)  # Add legend to identify data series
    # Add faint grid lines to aid readability
    axes[0].grid(alpha=0.3)  # Add faint grid lines for readability

    # Sort the array values in ascending order
    sorted_data = np.sort(series)  # Sort array values in ascending order
    # Create an evenly-spaced array of integer values
    emp_cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)  # Create integer array with even spacing
    # Matplotlib or Plotly figure/axis operation
    axes[1].step(sorted_data, emp_cdf, color="#CCB974", linewidth=1.5, label="Empirical CDF")  # Draw step function line (for CDF plots)
    # Draw a line on the plot axes
    axes[1].plot(x, cdf_vals, color="#C44E52", linewidth=2, linestyle="--",  # Draw a line on the axes
                 # Compute and store the result in label
                 label="Fitted Uniform CDF")  # Store result in label
    # Set the title text for this subplot
    axes[1].set_title("CDF — Empirical vs Fitted")  # Set the subplot title
    # Label the x-axis
    axes[1].set_xlabel(label_with_unit(col))  # Label the x-axis
    # Label the y-axis
    axes[1].set_ylabel(cdf_label())  # Label the y-axis
    # Add a legend showing what each line/bar represents
    axes[1].legend(fontsize=8)  # Add legend to identify data series
    # Add faint grid lines to aid readability
    axes[1].grid(alpha=0.3)  # Add faint grid lines for readability

    # Adjust subplot spacing to prevent label overlap
    plt.tight_layout()  # Prevent subplot labels from overlapping
    # Render the matplotlib figure inside the Streamlit app
    st.pyplot(fig)  # Render the matplotlib figure in Streamlit
    # Close the figure to free memory after rendering
    plt.close(fig)  # Close figure to free memory

    # Run Kolmogorov-Smirnov test to check goodness-of-fit to a distribution
    ks_stat, ks_p = sp.kstest(series, "uniform", args=(a, b-a))  # Kolmogorov-Smirnov goodness-of-fit test

    # Execute this operation
    _show_dist_stats("Uniform", {  # Execute this statement
        # Pandas data operation
        "n (sample size)": len(series),  # Execute this statement
        # Execute this operation
        "a — lower bound (min)": round(a, 4),  # Execute this statement
        # Execute this operation
        "b — upper bound (max)": round(b, 4),  # Execute this statement
        # Execute this operation
        "Mean ((a+b)/2)": round((a+b)/2, 4),  # Execute this statement
        # Execute this operation
        "Variance ((b-a)²/12)": round((b-a)**2/12, 4),  # Execute this statement
        # Execute this operation
        "Skewness (theoretical: 0)": 0,  # Execute this statement
        # Compute skewness — measure of distribution asymmetry
        "Skewness (data)": round(float(sp.skew(series)), 4),  # Compute skewness of the distribution
        # Execute this operation
        "Kurtosis excess (theoretical: -1.2)": -1.2,  # Execute this statement
        # Compute kurtosis — measure of distribution tail heaviness
        "Kurtosis excess (data)": round(float(sp.kurtosis(series)), 4),  # Compute kurtosis of the distribution
        # Execute this operation
        "KS statistic": round(ks_stat, 4),  # Execute this statement
        # Execute this operation
        "KS p-value": round(ks_p, 4),  # Execute this statement
    # Execute this operation
    })  # Execute this statement

    # Open a context manager (auto-closes when done)
    with st.expander("📖 Formula & Interpretation"):  # Open context manager
        # Render a mathematical formula using LaTeX notation
        st.latex(r"f(x) = \frac{1}{b - a}, \quad a \leq x \leq b")  # Render a LaTeX mathematical formula
        # Render a mathematical formula using LaTeX notation
        st.latex(r"\hat{a}_{MLE} = \min(x_i), \quad \hat{b}_{MLE} = \max(x_i)")  # Render a LaTeX mathematical formula
        # Check condition and branch accordingly
        if ks_p > 0.05:  # Check condition
            # Show a green success message to the user
            st.success(f"✅ KS p={ks_p:.4f} > 0.05 → data is consistent with Uniform distribution.")  # Show a green success message
        else:
            # Show a yellow warning message to the user
            st.warning(f"⚠️ KS p={ks_p:.4f} ≤ 0.05 → data does not fit Uniform well (common for real data).")  # Show a yellow warning message
        # Display text or data in the Streamlit UI
        st.write("Most real-world data is NOT uniform. If KS fails, it simply means the data has a non-uniform shape.")  # Display text or data in the Streamlit UI


# Define the _show_dist_stats function
def _show_dist_stats(dist_name: str, stats: dict):  # Define the _show_dist_stats function
    """Display distribution statistics table."""
    # Render formatted markdown text in the Streamlit UI
    st.markdown(f"#### 📋 {dist_name} — Estimated Parameters & Statistics")  # Render formatted markdown text in the Streamlit UI
    # Create a new DataFrame from a dictionary or array
    stats_df = pd.DataFrame({"Property": list(stats.keys()), "Value": list(stats.values())})  # Create DataFrame from dictionary or array
    # Apply a function to each element or row/column
    stats_df["Value"] = stats_df["Value"].apply(  # Store result in stats_df
        # Execute this operation
        lambda v: round(v, 4) if isinstance(v, float) else v  # Execute this statement
    )
    # Render an interactive data table in the UI
    st.dataframe(stats_df, use_container_width=True, hide_index=True)  # Render an interactive data table
