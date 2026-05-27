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
from utils.helpers import (          # Import shared utility functions
    get_numeric_columns,             # To list numeric columns
    drop_missing,                    # To clean NaN values
    section_header,                  # For styled section header
    label_with_unit,                 # Build axis label with unit
    density_label,                   # Standard density Y-axis label
    probability_label,               # Standard PMF Y-axis label
    render_inline_unit_input,        # Inline unit input widget
)


# Define the render_fitting function
def render_fitting(df: pd.DataFrame):  # Define the render_fitting function
    """
    # Execute this operation
    Main function for the Distribution Fitting module.  # Execute this statement
    # Execute this operation
    Allows the user to fit real data to theoretical distributions  # Execute this statement
    # Execute this operation
    and run the Central Limit Theorem simulation.  # Execute this statement

    # Execute this operation
    Parameters:  # Execute this statement
        # Create a new DataFrame from a dictionary or array
        df (pd.DataFrame): The uploaded dataset  # Create DataFrame from dictionary or array
    """
    section_header("Distribution Fitting & CLT", "🔗")  # Section heading

    # Sub-module selection: fitting or CLT simulation
    tab1, tab2 = st.tabs(["📐 Distribution Fitting", "📊 CLT Simulation"])  # Two tabs

    # Open a context manager (auto-closes when done)
    with tab1:  # Open context manager
        _render_fitting_tab(df)   # Render the fitting UI

    # Open a context manager (auto-closes when done)
    with tab2:  # Open context manager
        _render_clt_tab(df)       # Render the CLT simulation UI


# Define the _render_fitting_tab function
def _render_fitting_tab(df: pd.DataFrame):  # Define the _render_fitting_tab function
    """
    # Execute this operation
    Render the Distribution Fitting sub-module.  # Execute this statement
    # Execute this operation
    Fits selected data to multiple distributions and compares them.  # Execute this statement

    # Execute this operation
    Parameters:  # Execute this statement
        # Create a new DataFrame from a dictionary or array
        df (pd.DataFrame): The uploaded dataset  # Create DataFrame from dictionary or array
    """
    st.markdown("### Fit Data to a Theoretical Distribution")  # Sub-heading

    # Get numeric columns for user to choose from
    # Pandas data operation
    numeric_cols = get_numeric_columns(df)  # Store result in numeric_cols
    # Check condition and branch accordingly
    if not numeric_cols:  # Check condition
        # Show a red error message to the user
        st.error("No numeric columns found.")  # Show a red error message
        # Exit the function early — stop execution here
        return  # Exit function early

    # Let user select which column's data to fit
    # Compute and store the result in col
    col = st.selectbox("Select column to fit:", numeric_cols, key="fit_col")  # Column dropdown
    render_inline_unit_input(col, "fit")  # Unit input — shown on all chart axes for this column

    # Let user select which distribution to fit
    # Compute and store the result in dist_choice
    dist_choice = st.selectbox(  # Store result in dist_choice
        # Execute this operation
        "Select distribution to fit:",  # Execute this statement
        ["Normal", "Exponential", "Poisson", "Uniform"],  # Available distributions
        # Compute and store the result in key
        key="fit_dist"  # Store result in key
    )

    if st.button("Fit Distribution"):       # Trigger on button click
        series = drop_missing(df[col])      # Remove NaN values from the column

        # Need at least 10 data points for a meaningful fit
        # Check condition and branch accordingly
        if len(series) < 10:  # Check condition
            # Show a red error message to the user
            st.error("Need at least 10 data points to fit a distribution.")  # Show a red error message
            # Exit the function early — stop execution here
            return  # Exit function early

        # Call the appropriate fitting function based on user's choice
        # Check condition and branch accordingly
        if dist_choice == "Normal":  # Check condition
            # Pandas data operation
            _fit_normal(series, col)  # Execute this statement
        # Alternative condition check
        elif dist_choice == "Exponential":  # Check alternative condition
            # Pandas data operation
            _fit_exponential(series, col)  # Execute this statement
        # Alternative condition check
        elif dist_choice == "Poisson":  # Check alternative condition
            # Pandas data operation
            _fit_poisson(series, col)  # Execute this statement
        # Alternative condition check
        elif dist_choice == "Uniform":  # Check alternative condition
            # Pandas data operation
            _fit_uniform(series, col)  # Execute this statement


# Define the _fit_normal function
def _fit_normal(series: pd.Series, col_name: str):  # Define the _fit_normal function
    """
    # Execute this operation
    Fit a Normal distribution to the data and plot the result.  # Execute this statement
    # SciPy statistical computation
    Uses maximum likelihood estimation (MLE) via scipy.  # Execute this statement

    # Execute this operation
    Parameters:  # Execute this statement
        # Pandas data operation
        series (pd.Series): Cleaned numeric data  # Pandas data operation
        # Execute this operation
        col_name (str): Column name for display  # Execute this statement
    """
    # sp.norm.fit returns (mean, std) estimated from the data
    # Estimate Normal distribution parameters (μ, σ) from the data using MLE
    mu, sigma = sp.norm.fit(series)  # Fit Normal distribution parameters via MLE

    # Create x values spanning the range of the data
    # Create an evenly-spaced array of float values between two bounds
    x = np.linspace(series.min(), series.max(), 300)  # Create float array with even spacing

    # Compute the fitted PDF values at each x
    # Compute Normal probability density function values
    pdf_fitted = sp.norm.pdf(x, loc=mu, scale=sigma)  # Compute Normal PDF values

    # Create the comparison plot
    # Create a new figure with one or more subplot axes
    fig, ax = plt.subplots(figsize=(9, 5))  # Create figure and subplot axes
    ax.hist(series, bins=30, density=True,          # Actual data as histogram (normalized to density)
            # Compute and store the result in alpha
            alpha=0.5, color="#4C72B0", label="Data", edgecolor="white")  # Store result in alpha
    ax.plot(x, pdf_fitted, "r-", linewidth=2,       # Fitted curve in red
            # Compute and store the result in label
            label=f"Normal fit: μ={mu:.2f}, σ={sigma:.2f}")  # Store result in label
    ax.set_title(f"Normal Fit for '{col_name}'")    # Title
    # Label the x-axis
    ax.set_xlabel(label_with_unit(col_name))  # Label the x-axis
    # Label the y-axis
    ax.set_ylabel(density_label())  # Label the y-axis
    ax.legend()                                     # Show legend
    # Add faint grid lines to aid readability
    ax.grid(alpha=0.3)  # Add faint grid lines for readability
    # Render the matplotlib figure inside the Streamlit app
    st.pyplot(fig)  # Render the matplotlib figure in Streamlit
    # Close the figure to free memory after rendering
    plt.close(fig)  # Close figure to free memory

    # Goodness-of-fit test using Kolmogorov-Smirnov test
    ks_stat, ks_p = sp.kstest(series, "norm", args=(mu, sigma))  # KS test against fitted normal

    # Show the fitted parameters and test results
    # Open a context manager (auto-closes when done)
    with st.expander("📋 Fitting Results", expanded=True):  # Open context manager
        # Render formatted markdown text in the Streamlit UI
        st.markdown("**Estimated Parameters:**")  # Render formatted markdown text in the Streamlit UI
        # Display text or data in the Streamlit UI
        st.write(f"- Mean (μ): `{mu:.4f}`")  # Display text or data in the Streamlit UI
        # Display text or data in the Streamlit UI
        st.write(f"- Std Dev (σ): `{sigma:.4f}`")  # Display text or data in the Streamlit UI
        # Display text or data in the Streamlit UI
        st.write(f"- Variance (σ²): `{sigma**2:.4f}`")  # Display text or data in the Streamlit UI
        # Render formatted markdown text in the Streamlit UI
        st.markdown("**Goodness-of-Fit (Kolmogorov-Smirnov Test):**")  # Render formatted markdown text in the Streamlit UI
        # Display text or data in the Streamlit UI
        st.write(f"- KS Statistic: `{ks_stat:.4f}`")  # Display text or data in the Streamlit UI
        # Display text or data in the Streamlit UI
        st.write(f"- P-value: `{ks_p:.4f}`")  # Display text or data in the Streamlit UI
        # Interpret the KS test result
        # Check condition and branch accordingly
        if ks_p > 0.05:  # Check condition
            # Show a green success message to the user
            st.success("✅ Data is consistent with a Normal distribution (p > 0.05).")  # Show a green success message
        else:
            # Show a yellow warning message to the user
            st.warning("⚠️ Data does NOT fit a Normal distribution well (p ≤ 0.05).")  # Show a yellow warning message
        # Compute and display additional statistics
        # Render formatted markdown text in the Streamlit UI
        st.markdown("**Distribution Statistics:**")  # Render formatted markdown text in the Streamlit UI
        # Display text or data in the Streamlit UI
        st.write(f"- Skewness: `{sp.skew(series):.4f}`")  # Display text or data in the Streamlit UI
        # Display text or data in the Streamlit UI
        st.write(f"- Kurtosis (excess): `{sp.kurtosis(series):.4f}`")  # Display text or data in the Streamlit UI


# Define the _fit_exponential function
def _fit_exponential(series: pd.Series, col_name: str):  # Define the _fit_exponential function
    """
    # Execute this operation
    Fit an Exponential distribution to the data.  # Execute this statement
    # Execute this operation
    Only valid for non-negative data.  # Execute this statement

    # Execute this operation
    Parameters:  # Execute this statement
        # Pandas data operation
        series (pd.Series): Cleaned numeric data  # Pandas data operation
        # Execute this operation
        col_name (str): Column name for display  # Execute this statement
    """
    # Exponential distribution requires non-negative values
    # Check condition and branch accordingly
    if series.min() < 0:  # Check condition
        # Show a red error message to the user
        st.error("Exponential distribution requires non-negative values.")  # Show a red error message
        # Exit the function early — stop execution here
        return  # Exit function early

    # sp.expon.fit returns (loc, scale) — here loc≈0, scale=1/λ
    # Estimate Exponential distribution parameter λ from data using MLE
    loc, scale = sp.expon.fit(series)  # Fit Exponential distribution via MLE
    lam_est = 1 / scale          # Lambda = rate = 1/scale

    x = np.linspace(0, series.max(), 300)                # x from 0 to max
    pdf_fitted = sp.expon.pdf(x, loc=loc, scale=scale)   # Fitted PDF values

    # Create a new figure with one or more subplot axes
    fig, ax = plt.subplots(figsize=(9, 5))  # Create figure and subplot axes
    # Draw a histogram showing value frequency distribution
    ax.hist(series, bins=30, density=True, alpha=0.5,  # Draw histogram of the data
            # Compute and store the result in color
            color="#55A868", label="Data", edgecolor="white")  # Store result in color
    # Draw a line on the plot axes
    ax.plot(x, pdf_fitted, "r-", linewidth=2,  # Draw a line on the axes
            # Compute and store the result in label
            label=f"Exp fit: λ={lam_est:.3f}")  # Store result in label
    # Set the title text for this subplot
    ax.set_title(f"Exponential Fit for '{col_name}'")  # Set the subplot title
    # Label the x-axis
    ax.set_xlabel(label_with_unit(col_name))  # Label the x-axis
    # Label the y-axis
    ax.set_ylabel(density_label())  # Label the y-axis
    # Add a legend showing what each line/bar represents
    ax.legend()  # Add legend to identify data series
    # Add faint grid lines to aid readability
    ax.grid(alpha=0.3)  # Add faint grid lines for readability
    # Render the matplotlib figure inside the Streamlit app
    st.pyplot(fig)  # Render the matplotlib figure in Streamlit
    # Close the figure to free memory after rendering
    plt.close(fig)  # Close figure to free memory

    ks_stat, ks_p = sp.kstest(series, "expon", args=(loc, scale))  # KS goodness-of-fit

    # Open a context manager (auto-closes when done)
    with st.expander("📋 Fitting Results", expanded=True):  # Open context manager
        # Render formatted markdown text in the Streamlit UI
        st.markdown("**Estimated Parameters:**")  # Render formatted markdown text in the Streamlit UI
        # Display text or data in the Streamlit UI
        st.write(f"- Rate (λ): `{lam_est:.4f}`")  # Display text or data in the Streamlit UI
        # Display text or data in the Streamlit UI
        st.write(f"- Mean (1/λ): `{1/lam_est:.4f}`")  # Display text or data in the Streamlit UI
        # Display text or data in the Streamlit UI
        st.write(f"- Variance (1/λ²): `{1/lam_est**2:.4f}`")  # Display text or data in the Streamlit UI
        # Render formatted markdown text in the Streamlit UI
        st.markdown("**Goodness-of-Fit:**")  # Render formatted markdown text in the Streamlit UI
        # Display text or data in the Streamlit UI
        st.write(f"- KS Statistic: `{ks_stat:.4f}` | P-value: `{ks_p:.4f}`")  # Display text or data in the Streamlit UI
        # Check condition and branch accordingly
        if ks_p > 0.05:  # Check condition
            # Show a green success message to the user
            st.success("✅ Data is consistent with Exponential distribution (p > 0.05).")  # Show a green success message
        else:
            # Show a yellow warning message to the user
            st.warning("⚠️ Poor fit to Exponential distribution (p ≤ 0.05).")  # Show a yellow warning message
        # Display text or data in the Streamlit UI
        st.write(f"- Skewness: `{sp.skew(series):.4f}` (Exponential theoretical: 2.0)")  # Display text or data in the Streamlit UI
        # Display text or data in the Streamlit UI
        st.write(f"- Kurtosis: `{sp.kurtosis(series):.4f}` (Exponential theoretical: 6.0)")  # Display text or data in the Streamlit UI


# Define the _fit_poisson function
def _fit_poisson(series: pd.Series, col_name: str):  # Define the _fit_poisson function
    """
    # Execute this operation
    Fit a Poisson distribution to count data using MLE (mean estimate).  # Execute this statement

    # Execute this operation
    Parameters:  # Execute this statement
        # Pandas data operation
        series (pd.Series): Cleaned numeric data (should be non-negative integers)  # Pandas data operation
        # Execute this operation
        col_name (str): Column name for display  # Execute this statement
    """
    # For Poisson, the MLE estimate of λ is simply the sample mean
    lam_est = series.mean()   # Lambda = average count

    # x values are integer counts from 0 to max
    # Create an evenly-spaced array of integer values
    x = np.arange(0, int(series.max()) + 1)  # Create integer array with even spacing
    pmf_fitted = sp.poisson.pmf(x, mu=lam_est)  # Compute PMF values

    # Create a new figure with one or more subplot axes
    fig, ax = plt.subplots(figsize=(9, 5))  # Create figure and subplot axes
    # Compute relative frequency of each integer in the data
    # Count how many times each unique value appears
    value_counts = series.round().value_counts(normalize=True).sort_index()  # Store result in value_counts
    ax.bar(value_counts.index, value_counts.values,   # Actual frequencies
           # Compute and store the result in alpha
           alpha=0.5, color="#8172B2", label="Data", width=0.4, align="center")  # Store result in alpha
    ax.plot(x, pmf_fitted, "ro-", markersize=5,       # Fitted PMF as connected dots
            # Compute and store the result in linewidth
            linewidth=1.5, label=f"Poisson fit: λ={lam_est:.2f}")  # Store result in linewidth
    # Set the title text for this subplot
    ax.set_title(f"Poisson Fit for '{col_name}'")  # Set the subplot title
    # Label the x-axis
    ax.set_xlabel(label_with_unit(col_name))  # Label the x-axis
    # Label the y-axis
    ax.set_ylabel(probability_label())  # Label the y-axis
    # Add a legend showing what each line/bar represents
    ax.legend()  # Add legend to identify data series
    # Add faint grid lines to aid readability
    ax.grid(axis="y", alpha=0.3)  # Add faint grid lines for readability
    # Render the matplotlib figure inside the Streamlit app
    st.pyplot(fig)  # Render the matplotlib figure in Streamlit
    # Close the figure to free memory after rendering
    plt.close(fig)  # Close figure to free memory

    # Open a context manager (auto-closes when done)
    with st.expander("📋 Fitting Results", expanded=True):  # Open context manager
        # Render formatted markdown text in the Streamlit UI
        st.markdown("**Estimated Parameters:**")  # Render formatted markdown text in the Streamlit UI
        # Display text or data in the Streamlit UI
        st.write(f"- Lambda (λ): `{lam_est:.4f}`")  # Display text or data in the Streamlit UI
        # Display text or data in the Streamlit UI
        st.write(f"- Mean = Variance = λ: `{lam_est:.4f}`")  # Display text or data in the Streamlit UI
        # Display text or data in the Streamlit UI
        st.write(f"- Data Variance: `{series.var():.4f}` (should be ≈ λ for good fit)")  # Display text or data in the Streamlit UI
        # Display text or data in the Streamlit UI
        st.write(f"- Skewness: `{sp.skew(series):.4f}` (Poisson theoretical: {1/np.sqrt(lam_est):.4f})")  # Display text or data in the Streamlit UI


# Define the _fit_uniform function
def _fit_uniform(series: pd.Series, col_name: str):  # Define the _fit_uniform function
    """
    # Execute this operation
    Fit a Uniform distribution to the data using min and max as bounds.  # Execute this statement

    # Execute this operation
    Parameters:  # Execute this statement
        # Pandas data operation
        series (pd.Series): Cleaned numeric data  # Pandas data operation
        # Execute this operation
        col_name (str): Column name for display  # Execute this statement
    """
    a = series.min()   # Lower bound estimate = minimum value
    b = series.max()   # Upper bound estimate = maximum value

    x = np.linspace(a - 0.1*(b-a), b + 0.1*(b-a), 300)  # x slightly wider than [a, b]
    pdf_fitted = sp.uniform.pdf(x, loc=a, scale=b-a)      # Fitted PDF (constant between a and b)

    # Create a new figure with one or more subplot axes
    fig, ax = plt.subplots(figsize=(9, 5))  # Create figure and subplot axes
    # Draw a histogram showing value frequency distribution
    ax.hist(series, bins=30, density=True, alpha=0.5,  # Draw histogram of the data
            # Compute and store the result in color
            color="#CCB974", label="Data", edgecolor="white")  # Store result in color
    # Draw a line on the plot axes
    ax.plot(x, pdf_fitted, "r-", linewidth=2,  # Draw a line on the axes
            # Compute and store the result in label
            label=f"Uniform fit: [{a:.2f}, {b:.2f}]")  # Store result in label
    # Set the title text for this subplot
    ax.set_title(f"Uniform Fit for '{col_name}'")  # Set the subplot title
    # Label the x-axis
    ax.set_xlabel(label_with_unit(col_name))  # Label the x-axis
    # Label the y-axis
    ax.set_ylabel(density_label())  # Label the y-axis
    # Add a legend showing what each line/bar represents
    ax.legend()  # Add legend to identify data series
    # Add faint grid lines to aid readability
    ax.grid(alpha=0.3)  # Add faint grid lines for readability
    # Render the matplotlib figure inside the Streamlit app
    st.pyplot(fig)  # Render the matplotlib figure in Streamlit
    # Close the figure to free memory after rendering
    plt.close(fig)  # Close figure to free memory

    ks_stat, ks_p = sp.kstest(series, "uniform", args=(a, b-a))  # KS goodness-of-fit

    # Open a context manager (auto-closes when done)
    with st.expander("📋 Fitting Results", expanded=True):  # Open context manager
        # Display text or data in the Streamlit UI
        st.write(f"- Lower bound (a): `{a:.4f}`")  # Display text or data in the Streamlit UI
        # Display text or data in the Streamlit UI
        st.write(f"- Upper bound (b): `{b:.4f}`")  # Display text or data in the Streamlit UI
        # Display text or data in the Streamlit UI
        st.write(f"- Mean: `{(a+b)/2:.4f}`")  # Display text or data in the Streamlit UI
        # Display text or data in the Streamlit UI
        st.write(f"- Variance: `{(b-a)**2/12:.4f}`")  # Display text or data in the Streamlit UI
        # Display text or data in the Streamlit UI
        st.write(f"- KS Statistic: `{ks_stat:.4f}` | P-value: `{ks_p:.4f}`")  # Display text or data in the Streamlit UI
        # Check condition and branch accordingly
        if ks_p > 0.05:  # Check condition
            # Show a green success message to the user
            st.success("✅ Data is consistent with Uniform distribution.")  # Show a green success message
        else:
            # Show a yellow warning message to the user
            st.warning("⚠️ Poor fit to Uniform distribution.")  # Show a yellow warning message


# =====================================================================
# Central Limit Theorem (CLT) Simulation
# =====================================================================

# Define the _render_clt_tab function
def _render_clt_tab(df: pd.DataFrame):  # Define the _render_clt_tab function
    """
    # Execute this operation
    Render the Central Limit Theorem simulation.  # Execute this statement
    # Execute this operation
    Takes many random samples, computes their means,  # Execute this statement
    # Execute this operation
    and shows that the distribution of means is approximately Normal.  # Execute this statement

    # Execute this operation
    Parameters:  # Execute this statement
        # Create a new DataFrame from a dictionary or array
        df (pd.DataFrame): The uploaded dataset  # Create DataFrame from dictionary or array
    """
    st.markdown("### Central Limit Theorem (CLT) Simulation")  # Sub-heading

    # Explanation box
    # Show an informational blue message box
    st.info("""  # Show an informational blue message box
    # Execute this operation
    **The Central Limit Theorem states:**  # Execute this statement
    # Execute this operation
    If you take many random samples from *any* distribution,   # Execute this statement
    # Execute this operation
    the distribution of sample means will approach a Normal distribution,  # Execute this statement
    # Execute this operation
    regardless of the original data's shape.  # Execute this statement
    """)

    # Get numeric columns to choose from
    # Pandas data operation
    numeric_cols = get_numeric_columns(df)  # Store result in numeric_cols
    # Check condition and branch accordingly
    if not numeric_cols:  # Check condition
        # Show a red error message to the user
        st.error("No numeric columns found.")  # Show a red error message
        # Exit the function early — stop execution here
        return  # Exit function early

    # Compute and store the result in col
    col = st.selectbox("Select column for CLT simulation:", numeric_cols, key="clt_col")  # Column dropdown
    render_inline_unit_input(col, "clt")  # Unit input — shown on CLT chart axes

    # Let user configure the simulation parameters
    # Compute and store the result in c1, c2
    c1, c2 = st.columns(2)  # Store result in c1, c2
    # Open a context manager (auto-closes when done)
    with c1:  # Open context manager
        # Compute and store the result in sample_size
        sample_size = st.slider(  # Store result in sample_size
            # Execute this operation
            "Sample size (n) per draw:",  # Execute this statement
            # Compute and store the result in min_value
            min_value=2, max_value=500, value=30, key="clt_n"  # Store result in min_value
        )  # How many values in each sample
    # Open a context manager (auto-closes when done)
    with c2:  # Open context manager
        # Compute and store the result in num_samples
        num_samples = st.slider(  # Store result in num_samples
            # Execute this operation
            "Number of samples (repetitions):",  # Execute this statement
            # Compute and store the result in min_value
            min_value=50, max_value=5000, value=1000, step=50, key="clt_reps"  # Store result in min_value
        )  # How many times to draw a sample

    if st.button("Run CLT Simulation"):              # Trigger simulation
        series = drop_missing(df[col])               # Remove NaN values

        # Check condition and branch accordingly
        if len(series) < sample_size:  # Check condition
            # Cannot draw a sample larger than the data
            # Show a red error message to the user
            st.error(f"Dataset has only {len(series)} values — reduce sample size below that.")  # Show a red error message
            # Exit the function early — stop execution here
            return  # Exit function early

        # Draw `num_samples` random samples, each of size `sample_size`
        # Compute the mean of each sample
        np.random.seed(42)   # Fix random seed for reproducibility
        # Compute and store the result in sample_means
        sample_means = [  # Store result in sample_means
            # Compute the arithmetic mean (average) of the values
            np.mean(np.random.choice(series, size=sample_size, replace=True))  # Compute arithmetic mean
            # np.random.choice with replace=True allows bootstrap sampling
            for _ in range(num_samples)   # Repeat num_samples times
        ]
        sample_means = np.array(sample_means)   # Convert list to numpy array for operations

        # Create the CLT visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # Two subplots: original and sample means

        # LEFT: Original data distribution
        # Draw a histogram showing value frequency distribution
        axes[0].hist(series, bins=30, density=True, alpha=0.7,  # Draw histogram of the data
                     # Compute and store the result in color
                     color="#4C72B0", edgecolor="white")  # Store result in color
        # Set the title text for this subplot
        axes[0].set_title(f"Original Data Distribution\n'{col}'")  # Set the subplot title
        # Label the x-axis
        axes[0].set_xlabel(label_with_unit(col))  # Label the x-axis
        # Label the y-axis
        axes[0].set_ylabel(density_label())  # Label the y-axis
        # Add faint grid lines to aid readability
        axes[0].grid(alpha=0.3)  # Add faint grid lines for readability

        # RIGHT: Distribution of sample means
        # Draw a histogram showing value frequency distribution
        axes[1].hist(sample_means, bins=40, density=True, alpha=0.7,  # Draw histogram of the data
                     # Compute and store the result in color
                     color="#55A868", edgecolor="white", label="Sample Means")  # Store result in color

        # Overlay the theoretical normal curve predicted by CLT
        mu_clt = series.mean()                        # CLT predicted mean = population mean
        sigma_clt = series.std(ddof=1) / np.sqrt(sample_size)  # CLT predicted std = σ/√n
        x_norm = np.linspace(sample_means.min(), sample_means.max(), 300)  # x range for curve
        axes[1].plot(x_norm, sp.norm.pdf(x_norm, mu_clt, sigma_clt),      # Normal curve
                     # Compute and store the result in "r-", linewidth
                     "r-", linewidth=2.5, label="Theoretical Normal")  # Store result in "r-", linewidth
        # Set the title text for this subplot
        axes[1].set_title(f"Distribution of {num_samples} Sample Means\n(n={sample_size} each)")  # Set the subplot title
        # Label the x-axis
        axes[1].set_xlabel(f"Sample Mean of {col}")  # Label x-axis: clarify this is a mean of the selected column
        # Label the y-axis
        axes[1].set_ylabel(density_label())  # Label the y-axis
        # Add a legend showing what each line/bar represents
        axes[1].legend()  # Add legend to identify data series
        # Add faint grid lines to aid readability
        axes[1].grid(alpha=0.3)  # Add faint grid lines for readability

        plt.tight_layout()     # Prevent label overlap
        st.pyplot(fig)         # Render plot
        plt.close(fig)         # Free memory

        # Show CLT statistics
        # Open a context manager (auto-closes when done)
        with st.expander("📋 CLT Simulation Statistics", expanded=True):  # Open context manager
            # Render formatted markdown text in the Streamlit UI
            st.markdown("**Original Population:**")  # Render formatted markdown text in the Streamlit UI
            # Display text or data in the Streamlit UI
            st.write(f"- Mean (μ): `{series.mean():.4f}`")  # Display text or data in the Streamlit UI
            # Display text or data in the Streamlit UI
            st.write(f"- Std Dev (σ): `{series.std(ddof=1):.4f}`")  # Display text or data in the Streamlit UI

            # Render formatted markdown text in the Streamlit UI
            st.markdown("**Theoretical CLT Prediction for Sample Means:**")  # Render formatted markdown text in the Streamlit UI
            # Display text or data in the Streamlit UI
            st.write(f"- Expected mean of means: `{mu_clt:.4f}`")  # Display text or data in the Streamlit UI
            # Display text or data in the Streamlit UI
            st.write(f"- Expected std of means (σ/√n): `{sigma_clt:.4f}`")  # Display text or data in the Streamlit UI
            st.latex(r"\sigma_{\bar{x}} = \frac{\sigma}{\sqrt{n}}")  # CLT formula

            # Render formatted markdown text in the Streamlit UI
            st.markdown("**Observed Sample Means Distribution:**")  # Render formatted markdown text in the Streamlit UI
            # Display text or data in the Streamlit UI
            st.write(f"- Actual mean of means: `{sample_means.mean():.4f}`")  # Display text or data in the Streamlit UI
            # Display text or data in the Streamlit UI
            st.write(f"- Actual std of means: `{sample_means.std():.4f}`")  # Display text or data in the Streamlit UI
            # Display text or data in the Streamlit UI
            st.write(f"- Skewness of means: `{sp.skew(sample_means):.4f}` (should be ≈ 0)")  # Display text or data in the Streamlit UI
            # Display text or data in the Streamlit UI
            st.write(f"- Kurtosis of means: `{sp.kurtosis(sample_means):.4f}` (should be ≈ 0)")  # Display text or data in the Streamlit UI

            # Show a green success message to the user
            st.success(  # Show a green success message
                # Compute and store the result in f"✅ With n
                f"✅ With n={sample_size}, the distribution of sample means is approximately Normal. "  # Store result in f"✅ With n
                # Execute this operation
                f"This demonstrates the Central Limit Theorem."  # Execute this statement
            )
