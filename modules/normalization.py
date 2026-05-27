# =========================
# modules/normalization.py
# Normalization module
# Implements Min-Max scaling and Z-score normalization
# Shows formulas, before/after comparison, and download option
# =========================

import pandas as pd               # Import pandas for DataFrame operations
import numpy as np                # Import numpy for mathematical calculations
import matplotlib.pyplot as plt   # Import matplotlib for comparison plots
import streamlit as st            # Import streamlit for UI rendering
from utils.helpers import (       # Import shared utility functions
    get_numeric_columns,          # To list numeric columns for selection
    drop_missing,                 # To handle NaN values before normalization
    section_header,               # For styled section header
)


def render_normalization(df: pd.DataFrame):
    """
    Main function for the Normalization module.
    Renders the full normalization UI inside the Streamlit app.

    Parameters:
        df (pd.DataFrame): The uploaded dataset
    """
    section_header("Data Normalization", "⚖️")  # Display section heading

    # Get all numeric columns available for normalization
    numeric_cols = get_numeric_columns(df)

    # If there are no numeric columns, stop
    if not numeric_cols:
        st.error("No numeric columns found in the dataset.")
        return  # Exit early

    # Let user select which columns to normalize (multi-select)
    selected_cols = st.multiselect(
        "Select columns to normalize:",
        numeric_cols,                                        # Options
        default=numeric_cols[:min(3, len(numeric_cols))],   # Default: first 3 columns
        key="norm_cols"
    )

    # Stop if the user hasn't selected any columns
    if not selected_cols:
        st.warning("Please select at least one column.")
        return

    # Let user choose the normalization method
    method = st.radio(
        "Select normalization method:",
        ["Min-Max Scaling", "Z-score Normalization"],  # Two methods available
        key="norm_method"
    )

    # Trigger normalization on button click
    if st.button("Apply Normalization"):
        if method == "Min-Max Scaling":
            _minmax_normalization(df, selected_cols)   # Call Min-Max function
        elif method == "Z-score Normalization":
            _zscore_normalization(df, selected_cols)   # Call Z-score function


# =====================================================================
# Normalization helper functions below
# =====================================================================

def _minmax_normalization(df: pd.DataFrame, cols: list):
    """
    Apply Min-Max normalization to selected columns.
    Scales all values to the range [0, 1].

    Formula:
        x_norm = (x - x_min) / (x_max - x_min)

    Parameters:
        df (pd.DataFrame): Full dataset
        cols (list): List of column names to normalize
    """
    # Create a copy so we don't modify the original DataFrame
    result_df = df[cols].copy()

    # Apply Min-Max formula to each selected column
    for col in cols:
        series = drop_missing(df[col])            # Remove NaN values for clean computation
        x_min = series.min()                      # Minimum value in column
        x_max = series.max()                      # Maximum value in column

        # Guard against division by zero (happens when all values are the same)
        if x_max == x_min:
            result_df[col] = 0.0                  # If all values are equal, set normalized to 0
            st.warning(f"Column '{col}' has zero range — all values are the same.")
            continue  # Skip to next column

        # Apply the Min-Max formula: (x - min) / (max - min)
        result_df[col] = (df[col] - x_min) / (x_max - x_min)

    # Display results
    with st.expander("✅ Min-Max Normalization Results", expanded=True):
        # Show the formula using LaTeX
        st.latex(r"x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}")

        # Explain what the formula means
        st.info("Min-Max scaling compresses all values into [0, 1]. Min → 0, Max → 1.")

        # Show before and after tables side by side
        c1, c2 = st.columns(2)           # Create two columns in the UI
        with c1:
            st.markdown("**Before Normalization:**")   # Label
            st.dataframe(df[cols].head(10), use_container_width=True)  # Original data
        with c2:
            st.markdown("**After Normalization:**")    # Label
            st.dataframe(result_df.head(10), use_container_width=True)  # Normalized data

        # Comparison statistics (min and max should now be 0 and 1)
        st.markdown("**Normalized Column Statistics:**")
        st.dataframe(result_df.describe().T, use_container_width=True)  # .T = transpose for readability

        # Plot before vs. after for the first selected column
        _plot_before_after(df[cols[0]], result_df[cols[0]], cols[0], "Min-Max")

        # Provide download button for the normalized data
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Normalized CSV", csv, "minmax_normalized.csv", "text/csv")


def _zscore_normalization(df: pd.DataFrame, cols: list):
    """
    Apply Z-score (standardization) normalization to selected columns.
    Transforms values so the column has mean=0 and std=1.

    Formula:
        z = (x - μ) / σ

    Parameters:
        df (pd.DataFrame): Full dataset
        cols (list): List of column names to normalize
    """
    # Create a copy to store normalized values
    result_df = df[cols].copy()

    # Apply Z-score formula to each selected column
    for col in cols:
        series = drop_missing(df[col])          # Remove NaN values
        mu = series.mean()                      # Compute the mean (μ) of the column
        sigma = series.std(ddof=1)              # Compute the standard deviation (σ), sample version

        # Guard against zero standard deviation (all values are the same)
        if sigma == 0:
            result_df[col] = 0.0                # If std is 0, z-score is undefined — set to 0
            st.warning(f"Column '{col}' has zero std deviation — all values are the same.")
            continue  # Skip to next column

        # Apply Z-score formula: (x - mean) / std
        result_df[col] = (df[col] - mu) / sigma

    # Display results
    with st.expander("✅ Z-score Normalization Results", expanded=True):
        # Show the formula using LaTeX
        st.latex(r"z = \frac{x - \mu}{\sigma}")

        # Explain the formula
        st.info("Z-score normalization centers data at 0 with std=1. Useful for comparing different scales.")

        # Before/After table comparison
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Before Normalization:**")
            st.dataframe(df[cols].head(10), use_container_width=True)   # Original
        with c2:
            st.markdown("**After Normalization:**")
            st.dataframe(result_df.head(10), use_container_width=True)  # Z-scored

        # Show statistics — mean should be ~0, std should be ~1
        st.markdown("**Normalized Column Statistics (mean≈0, std≈1):**")
        st.dataframe(result_df.describe().T, use_container_width=True)

        # Comparison plot
        _plot_before_after(df[cols[0]], result_df[cols[0]], cols[0], "Z-score")

        # Download normalized data
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Normalized CSV", csv, "zscore_normalized.csv", "text/csv")


def _plot_before_after(original: pd.Series, normalized: pd.Series, col_name: str, method: str):
    """
    Plot side-by-side histograms comparing a column before and after normalization.

    Parameters:
        original (pd.Series): Original column values
        normalized (pd.Series): Normalized column values
        col_name (str): Column name for plot title
        method (str): Name of the normalization method used
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # Create 1 row, 2 columns of subplots

    # Left plot: original distribution
    axes[0].hist(original.dropna(), bins=30, color="#4C72B0", edgecolor="white", alpha=0.8)
    axes[0].set_title(f"Before: {col_name}")     # Title for original
    axes[0].set_xlabel("Value")                  # x-axis label
    axes[0].set_ylabel("Frequency")              # y-axis label
    axes[0].grid(axis="y", alpha=0.3)            # Faint grid

    # Right plot: normalized distribution
    axes[1].hist(normalized.dropna(), bins=30, color="#55A868", edgecolor="white", alpha=0.8)
    axes[1].set_title(f"After {method}: {col_name}")  # Title for normalized
    axes[1].set_xlabel("Normalized Value")             # x-axis label
    axes[1].set_ylabel("Frequency")                   # y-axis label
    axes[1].grid(axis="y", alpha=0.3)                 # Faint grid

    plt.tight_layout()                  # Prevent overlapping titles
    st.pyplot(fig)                      # Render in Streamlit
    plt.close(fig)                      # Free memory after rendering
