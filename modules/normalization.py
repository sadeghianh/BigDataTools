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
    label_with_unit,              # Build axis labels with unit e.g. "Salary ($)"
    freq_label,                   # Standard Y-axis label for frequency histograms
    render_inline_unit_input,     # Inline unit input widget next to column selector
)


# Define the render_normalization function
def render_normalization(df: pd.DataFrame):  # Define the render_normalization function
    """
    # Execute this operation
    Main function for the Normalization module.  # Execute this statement
    # Execute this operation
    Renders the full normalization UI inside the Streamlit app.  # Execute this statement

    # Execute this operation
    Parameters:  # Execute this statement
        # Create a new DataFrame from a dictionary or array
        df (pd.DataFrame): The uploaded dataset  # Create DataFrame from dictionary or array
    """
    section_header("Data Normalization", "⚖️")  # Display section heading

    # Get all numeric columns available for normalization
    # Pandas data operation
    numeric_cols = get_numeric_columns(df)  # Store result in numeric_cols

    # If there are no numeric columns, stop
    # Check condition and branch accordingly
    if not numeric_cols:  # Check condition
        # Show a red error message to the user
        st.error("No numeric columns found in the dataset.")  # Show a red error message
        return  # Exit early

    # Let user select which columns to normalize (multi-select)
    # Compute and store the result in selected_cols
    selected_cols = st.multiselect(  # Store result in selected_cols
        # Execute this operation
        "Select columns to normalize:",  # Execute this statement
        numeric_cols,                                        # Options
        default=numeric_cols[:min(3, len(numeric_cols))],   # Default: first 3 columns
        # Compute and store the result in key
        key="norm_cols"  # Store result in key
    )

    # Stop if the user hasn't selected any columns
    # Check condition and branch accordingly
    if not selected_cols:  # Check condition
        # Show a yellow warning message to the user
        st.warning("Please select at least one column.")  # Show a yellow warning message
        # Exit the function early — stop execution here
        return  # Exit function early

    # Let user define units for each selected column
    # These units appear on chart axes so viewers know what the numbers represent
    st.markdown("**📏 Column Units** *(optional — shown on chart axes)*")  # Section label
    unit_cols = st.columns(len(selected_cols))  # One input column per selected data column
    for i, col in enumerate(selected_cols):  # Loop over each selected column
        with unit_cols[i]:  # Place input in its column
            existing = st.session_state.get("col_units", {}).get(col, "")  # Get existing unit if any
            unit_val = st.text_input(  # Text input for this column's unit
                f"Unit for {col}:",       # Label
                value=existing,           # Pre-fill with any previously entered unit
                placeholder="e.g. $, kg",  # Hint
                key=f"norm_unit_{col}"    # Unique key
            )
            # Save entered unit to session state so all modules can use it
            if "col_units" not in st.session_state:  # Initialize dict if needed
                st.session_state["col_units"] = {}   # Create empty dict
            st.session_state["col_units"][col] = unit_val.strip()  # Store the unit

    # Let user choose the normalization method
    # Compute and store the result in method
    method = st.radio(  # Store result in method
        # Execute this operation
        "Select normalization method:",  # Execute this statement
        ["Min-Max Scaling", "Z-score Normalization"],  # Two methods available
        # Compute and store the result in key
        key="norm_method"  # Store result in key
    )

    # Trigger normalization on button click
    # Check condition and branch accordingly
    if st.button("Apply Normalization"):  # Check condition
        # Check condition and branch accordingly
        if method == "Min-Max Scaling":  # Check condition
            _minmax_normalization(df, selected_cols)   # Call Min-Max function
        # Alternative condition check
        elif method == "Z-score Normalization":  # Check alternative condition
            _zscore_normalization(df, selected_cols)   # Call Z-score function


# =====================================================================
# Normalization helper functions below
# =====================================================================

# Define the _minmax_normalization function
def _minmax_normalization(df: pd.DataFrame, cols: list):  # Define the _minmax_normalization function
    """
    # Execute this operation
    Apply Min-Max normalization to selected columns.  # Execute this statement
    # Execute this operation
    Scales all values to the range [0, 1].  # Execute this statement

    # Execute this operation
    Formula:  # Execute this statement
        # Compute and store the result in x_norm
        x_norm = (x - x_min) / (x_max - x_min)  # Store result in x_norm

    # Execute this operation
    Parameters:  # Execute this statement
        # Create a new DataFrame from a dictionary or array
        df (pd.DataFrame): Full dataset  # Create DataFrame from dictionary or array
        # Execute this operation
        cols (list): List of column names to normalize  # Execute this statement
    """
    # Create a copy so we don't modify the original DataFrame
    # Create a copy so the original DataFrame is not modified
    result_df = df[cols].copy()  # Store result in result_df

    # Apply Min-Max formula to each selected column
    # Iterate over each item in the sequence
    for col in cols:  # Loop over each item
        series = drop_missing(df[col])            # Remove NaN values for clean computation
        x_min = series.min()                      # Minimum value in column
        x_max = series.max()                      # Maximum value in column

        # Guard against division by zero (happens when all values are the same)
        # Check condition and branch accordingly
        if x_max == x_min:  # Check condition
            result_df[col] = 0.0                  # If all values are equal, set normalized to 0
            # Show a yellow warning message to the user
            st.warning(f"Column '{col}' has zero range — all values are the same.")  # Show a yellow warning message
            continue  # Skip to next column

        # Apply the Min-Max formula: (x - min) / (max - min)
        # Pandas data operation
        result_df[col] = (df[col] - x_min) / (x_max - x_min)  # Store result in result_df

    # Display results
    # Open a context manager (auto-closes when done)
    with st.expander("✅ Min-Max Normalization Results", expanded=True):  # Open context manager
        # Show the formula using LaTeX
        # Render a mathematical formula using LaTeX notation
        st.latex(r"x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}")  # Render a LaTeX mathematical formula

        # Explain what the formula means
        # Show an informational blue message box
        st.info("Min-Max scaling compresses all values into [0, 1]. Min → 0, Max → 1.")  # Show an informational blue message box

        # Show before and after tables side by side
        c1, c2 = st.columns(2)           # Create two columns in the UI
        # Open a context manager (auto-closes when done)
        with c1:  # Open context manager
            st.markdown("**Before Normalization:**")   # Label
            st.dataframe(df[cols].head(10), use_container_width=True)  # Original data
        # Open a context manager (auto-closes when done)
        with c2:  # Open context manager
            st.markdown("**After Normalization:**")    # Label
            st.dataframe(result_df.head(10), use_container_width=True)  # Normalized data

        # Comparison statistics (min and max should now be 0 and 1)
        # Render formatted markdown text in the Streamlit UI
        st.markdown("**Normalized Column Statistics:**")  # Render formatted markdown text in the Streamlit UI
        st.dataframe(result_df.describe().T, use_container_width=True)  # .T = transpose for readability

        # Plot before vs. after for the first selected column
        # Pandas data operation
        _plot_before_after(df[cols[0]], result_df[cols[0]], cols[0], "Min-Max")  # Execute this statement

        # Provide download button for the normalized data
        # Pandas data operation
        csv = result_df.to_csv(index=False).encode("utf-8")  # Store result in csv
        # Create a button for downloading a file
        st.download_button("⬇️ Download Normalized CSV", csv, "minmax_normalized.csv", "text/csv")  # Create a file download button


# Define the _zscore_normalization function
def _zscore_normalization(df: pd.DataFrame, cols: list):  # Define the _zscore_normalization function
    """
    # Execute this operation
    Apply Z-score (standardization) normalization to selected columns.  # Execute this statement
    # Compute and store the result in Transforms values so the column has mean
    Transforms values so the column has mean=0 and std=1.  # Store result in Transforms values so the column has mean

    # Execute this operation
    Formula:  # Execute this statement
        # Compute and store the result in z
        z = (x - μ) / σ  # Store result in z

    # Execute this operation
    Parameters:  # Execute this statement
        # Create a new DataFrame from a dictionary or array
        df (pd.DataFrame): Full dataset  # Create DataFrame from dictionary or array
        # Execute this operation
        cols (list): List of column names to normalize  # Execute this statement
    """
    # Create a copy to store normalized values
    # Create a copy so the original DataFrame is not modified
    result_df = df[cols].copy()  # Store result in result_df

    # Apply Z-score formula to each selected column
    # Iterate over each item in the sequence
    for col in cols:  # Loop over each item
        series = drop_missing(df[col])          # Remove NaN values
        mu = series.mean()                      # Compute the mean (μ) of the column
        sigma = series.std(ddof=1)              # Compute the standard deviation (σ), sample version

        # Guard against zero standard deviation (all values are the same)
        # Check condition and branch accordingly
        if sigma == 0:  # Check condition
            result_df[col] = 0.0                # If std is 0, z-score is undefined — set to 0
            # Show a yellow warning message to the user
            st.warning(f"Column '{col}' has zero std deviation — all values are the same.")  # Show a yellow warning message
            continue  # Skip to next column

        # Apply Z-score formula: (x - mean) / std
        # Pandas data operation
        result_df[col] = (df[col] - mu) / sigma  # Store result in result_df

    # Display results
    # Open a context manager (auto-closes when done)
    with st.expander("✅ Z-score Normalization Results", expanded=True):  # Open context manager
        # Show the formula using LaTeX
        # Render a mathematical formula using LaTeX notation
        st.latex(r"z = \frac{x - \mu}{\sigma}")  # Render a LaTeX mathematical formula

        # Explain the formula
        # Show an informational blue message box
        st.info("Z-score normalization centers data at 0 with std=1. Useful for comparing different scales.")  # Show an informational blue message box

        # Before/After table comparison
        # Compute and store the result in c1, c2
        c1, c2 = st.columns(2)  # Store result in c1, c2
        # Open a context manager (auto-closes when done)
        with c1:  # Open context manager
            # Render formatted markdown text in the Streamlit UI
            st.markdown("**Before Normalization:**")  # Render formatted markdown text in the Streamlit UI
            st.dataframe(df[cols].head(10), use_container_width=True)   # Original
        # Open a context manager (auto-closes when done)
        with c2:  # Open context manager
            # Render formatted markdown text in the Streamlit UI
            st.markdown("**After Normalization:**")  # Render formatted markdown text in the Streamlit UI
            st.dataframe(result_df.head(10), use_container_width=True)  # Z-scored

        # Show statistics — mean should be ~0, std should be ~1
        # Render formatted markdown text in the Streamlit UI
        st.markdown("**Normalized Column Statistics (mean≈0, std≈1):**")  # Render formatted markdown text in the Streamlit UI
        # Render an interactive data table in the UI
        st.dataframe(result_df.describe().T, use_container_width=True)  # Render an interactive data table

        # Comparison plot
        # Pandas data operation
        _plot_before_after(df[cols[0]], result_df[cols[0]], cols[0], "Z-score")  # Execute this statement

        # Download normalized data
        # Pandas data operation
        csv = result_df.to_csv(index=False).encode("utf-8")  # Store result in csv
        # Create a button for downloading a file
        st.download_button("⬇️ Download Normalized CSV", csv, "zscore_normalized.csv", "text/csv")  # Create a file download button


# Define the _plot_before_after function
def _plot_before_after(original: pd.Series, normalized: pd.Series, col_name: str, method: str):  # Define the _plot_before_after function
    """
    # Execute this operation
    Plot side-by-side histograms comparing a column before and after normalization.  # Execute this statement

    # Execute this operation
    Parameters:  # Execute this statement
        # Pandas data operation
        original (pd.Series): Original column values  # Pandas data operation
        # Pandas data operation
        normalized (pd.Series): Normalized column values  # Pandas data operation
        # Execute this operation
        col_name (str): Column name for plot title  # Execute this statement
        # Execute this operation
        method (str): Name of the normalization method used  # Execute this statement
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # Create 1 row, 2 columns of subplots

    # Left plot: original distribution
    # Draw a histogram showing value frequency distribution
    axes[0].hist(original.dropna(), bins=30, color="#4C72B0", edgecolor="white", alpha=0.8)  # Draw histogram of the data
    axes[0].set_title(f"Before: {col_name}")                    # Title for original
    axes[0].set_xlabel(label_with_unit(col_name))               # x-axis: original column values with unit
    axes[0].set_ylabel(freq_label())                            # y-axis: count of observations
    axes[0].grid(axis="y", alpha=0.3)                           # Faint grid

    # Right plot: normalized distribution
    # Draw a histogram showing value frequency distribution
    axes[1].hist(normalized.dropna(), bins=30, color="#55A868", edgecolor="white", alpha=0.8)  # Draw histogram of the data
    axes[1].set_title(f"After {method}: {col_name}")            # Title for normalized
    axes[1].set_xlabel(f"Normalized {col_name} (dimensionless — no unit after scaling)")  # x-axis: normalized values lose their original unit
    axes[1].set_ylabel(freq_label())                            # y-axis: count of observations
    axes[1].grid(axis="y", alpha=0.3)                 # Faint grid

    plt.tight_layout()                  # Prevent overlapping titles
    st.pyplot(fig)                      # Render in Streamlit
    plt.close(fig)                      # Free memory after rendering
