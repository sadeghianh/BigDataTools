# =========================
# utils/helpers.py
# Shared utility functions used across all modules
# =========================

import pandas as pd          # Import pandas for DataFrame operations
import numpy as np           # Import numpy for numerical checks
import streamlit as st       # Import streamlit for UI error display


def get_numeric_columns(df: pd.DataFrame) -> list:
    """
    Return a list of column names that contain numeric data.
    This is used to filter columns suitable for statistical analysis.

    Parameters:
        df (pd.DataFrame): The uploaded dataset

    Returns:
        list: Column names with numeric dtype
    """
    # Use pandas select_dtypes to filter only numeric columns (int, float)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols  # Return the list of numeric column names


def get_categorical_columns(df: pd.DataFrame) -> list:
    """
    Return a list of column names that contain categorical/text data.
    Used for stratified sampling and chi-square tests.

    Parameters:
        df (pd.DataFrame): The uploaded dataset

    Returns:
        list: Column names with object or category dtype
    """
    # Use select_dtypes to find columns of type 'object' or 'category'
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return cat_cols  # Return the list of categorical column names


def validate_dataframe(df) -> bool:
    """
    Check whether a DataFrame is valid and non-empty.
    Returns True if valid, False otherwise.
    Also shows a Streamlit warning if the dataset is empty.

    Parameters:
        df: Any value (should be a DataFrame)

    Returns:
        bool: True if valid DataFrame with rows, False otherwise
    """
    # Check if df is actually a pandas DataFrame (not None or wrong type)
    if not isinstance(df, pd.DataFrame):
        st.warning("⚠️ No dataset loaded. Please upload a CSV file first.")  # Show warning in UI
        return False  # Return False so calling code stops execution

    # Check if the DataFrame has at least one row of data
    if df.empty:
        st.warning("⚠️ The uploaded dataset is empty.")  # Inform the user the file has no rows
        return False  # Return False to prevent operations on empty data

    return True  # All checks passed — dataset is usable


def drop_missing(series: pd.Series) -> pd.Series:
    """
    Remove NaN (missing) values from a pandas Series.
    Used before statistical calculations to avoid errors.

    Parameters:
        series (pd.Series): A column from the dataset

    Returns:
        pd.Series: The same column with NaN values removed
    """
    # dropna() removes any NaN entries from the Series
    return series.dropna()


def safe_convert_numeric(series: pd.Series) -> pd.Series:
    """
    Attempt to convert a Series to numeric values.
    Non-convertible values become NaN instead of causing crashes.

    Parameters:
        series (pd.Series): Any column from the dataset

    Returns:
        pd.Series: Numeric Series with NaN for unconvertible entries
    """
    # pd.to_numeric with errors='coerce' replaces bad values with NaN instead of raising error
    return pd.to_numeric(series, errors="coerce")


def format_pvalue(p: float) -> str:
    """
    Format a p-value for clean display in the UI.
    Very small values are shown in scientific notation.

    Parameters:
        p (float): The p-value from a statistical test

    Returns:
        str: Human-readable string representation
    """
    # If p-value is extremely small (< 0.0001), use scientific notation
    if p < 0.0001:
        return f"{p:.4e}"  # Format as e.g. 1.2345e-07
    # Otherwise show up to 4 decimal places
    return f"{p:.4f}"


def section_header(title: str, emoji: str = "📌"):
    """
    Display a styled section header in the Streamlit UI.

    Parameters:
        title (str): The section title text
        emoji (str): An emoji prefix (default: 📌)
    """
    st.markdown("---")                          # Draw a horizontal divider line
    st.subheader(f"{emoji} {title}")            # Display the title as a subheader with emoji
