# =========================
# utils/helpers.py
# Shared utility functions used across all modules
# Added: unit management system so every chart axis shows the correct unit
# =========================

import pandas as pd          # Import pandas for DataFrame operations
import numpy as np           # Import numpy for numerical checks
import streamlit as st       # Import streamlit for UI error display


# =====================================================================
# UNIT MANAGEMENT SYSTEM
# Allows users to define units for each column once (in sidebar or any module)
# and have those units appear automatically on all chart axes and results
# =====================================================================

def render_unit_manager(df: pd.DataFrame):
    """
    Render a unit input section in the sidebar.
    User types the unit for each numeric column (e.g. $, kg, years, %).
    Units are stored in st.session_state["col_units"] so all modules can access them.

    Parameters:
        df (pd.DataFrame): The uploaded dataset — used to get column names
    """
    if not isinstance(df, pd.DataFrame) or df.empty:  # Only show if dataset is loaded
        return  # Exit if no dataset

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # Get numeric columns
    if not numeric_cols:  # No numeric columns — nothing to configure
        return

    # Initialize the units dictionary in session state if not already done
    if "col_units" not in st.session_state:  # First time loading
        st.session_state["col_units"] = {}  # Create empty dict to store units

    st.sidebar.markdown("---")  # Divider in sidebar
    st.sidebar.markdown("### 📏 Column Units")  # Section heading in sidebar
    st.sidebar.caption("Define units for numeric columns. They will appear on all chart axes.")

    # Show a text input for each numeric column
    for col in numeric_cols:  # Loop over every numeric column
        current = st.session_state["col_units"].get(col, "")  # Get existing unit or empty string
        unit = st.sidebar.text_input(
            f"{col}:",                          # Label = column name
            value=current,                      # Pre-fill with any previously entered unit
            placeholder="e.g. $, kg, years",   # Hint text
            key=f"unit_input_{col}"             # Unique key per column
        )
        st.session_state["col_units"][col] = unit.strip()  # Save the entered unit (stripped of spaces)


def get_unit(col: str) -> str:
    """
    Get the unit for a specific column from session state.
    Returns empty string if no unit was defined for this column.

    Usage in any module:
        unit = get_unit("Salary")  # Returns "$" if user entered "$"
        ax.set_xlabel(f"Salary ({unit})" if unit else "Salary")

    Parameters:
        col (str): Column name to look up

    Returns:
        str: The unit string (e.g. "$", "kg") or "" if not set
    """
    units = st.session_state.get("col_units", {})  # Get the units dict from session state
    return units.get(col, "")  # Return unit for this column, or empty string


def label_with_unit(col: str, unit: str = None) -> str:
    """
    Build a clean axis label combining column name and unit.
    If unit is None, looks it up automatically from session state.
    If unit is empty string, returns just the column name.

    Examples:
        label_with_unit("Salary", "$")    → "Salary ($)"
        label_with_unit("Age", "years")   → "Age (years)"
        label_with_unit("Score", "")      → "Score"
        label_with_unit("Salary")         → "Salary ($)"  (if $ was saved)

    Parameters:
        col (str): Column name
        unit (str, optional): Unit string. If None, auto-fetched from session state.

    Returns:
        str: Formatted label like "Salary ($)" or just "Salary"
    """
    if unit is None:  # If no unit passed, look it up from session state
        unit = get_unit(col)  # Fetch from the stored units dict
    if unit:  # Only add parentheses if there is actually a unit
        return f"{col} ({unit})"  # Format: "Salary ($)"
    return col  # No unit — just return column name as-is


def freq_label() -> str:
    """Return the standard Y-axis label for frequency/count histograms."""
    return "Frequency (number of observations)"  # Clear label explaining what Y axis shows


def density_label() -> str:
    """Return the standard Y-axis label for density/KDE plots."""
    return "Density (probability per unit)"  # Label for probability density plots


def probability_label() -> str:
    """Return the standard Y-axis label for PMF/probability plots."""
    return "Probability P(X = k)"  # Label for discrete probability mass function plots


def cdf_label() -> str:
    """Return the standard Y-axis label for CDF plots."""
    return "Cumulative Probability F(x) = P(X ≤ x)"  # Label for cumulative distribution function


# =====================================================================
# Existing helper functions (unchanged)
# =====================================================================

def get_numeric_columns(df: pd.DataFrame) -> list:
    """
    Return a list of column names that contain numeric data.
    Used to filter columns suitable for statistical analysis.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # Filter numeric columns
    return numeric_cols  # Return list of numeric column names


def get_categorical_columns(df: pd.DataFrame) -> list:
    """
    Return a list of column names that contain categorical/text data.
    Used for stratified sampling and chi-square tests.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()  # Filter categorical columns
    return cat_cols  # Return list of categorical column names


def validate_dataframe(df) -> bool:
    """
    Check whether a DataFrame is valid and non-empty.
    Returns True if valid, False otherwise.
    Also shows a Streamlit warning if the dataset is empty or missing.
    """
    if not isinstance(df, pd.DataFrame):  # Check if it's actually a DataFrame
        st.warning("⚠️ No dataset loaded. Please upload a CSV file first.")  # Warn user
        return False  # Invalid — stop processing

    if df.empty:  # Check if DataFrame has any rows
        st.warning("⚠️ The uploaded dataset is empty.")  # Warn user
        return False  # Empty — stop processing

    return True  # All checks passed — dataset is usable


def drop_missing(series: pd.Series) -> pd.Series:
    """
    Remove NaN (missing) values from a pandas Series.
    Used before statistical calculations to avoid errors.
    """
    return series.dropna()  # Remove NaN values and return clean series


def safe_convert_numeric(series: pd.Series) -> pd.Series:
    """
    Attempt to convert a Series to numeric values.
    Non-convertible values become NaN instead of causing crashes.
    """
    return pd.to_numeric(series, errors="coerce")  # Convert to numeric, bad values → NaN


def format_pvalue(p: float) -> str:
    """
    Format a p-value for clean display in the UI.
    Very small values are shown in scientific notation.
    """
    if p < 0.0001:  # Very small p-value — use scientific notation
        return f"{p:.4e}"  # e.g. 1.2345e-07
    return f"{p:.4f}"  # Standard 4 decimal places


def section_header(title: str, emoji: str = "📌"):
    """
    Display a styled section header in the Streamlit UI.
    """
    st.markdown("---")                    # Horizontal divider line
    st.subheader(f"{emoji} {title}")      # Section title with emoji


def render_inline_unit_input(col: str, key_suffix: str = "") -> str:
    """
    Render a small inline unit text input next to a column selector.
    Saves the entered unit to session_state["col_units"] automatically.
    Returns the unit string so callers can use it immediately.

    Usage:
        col = st.selectbox("Column:", numeric_cols)
        unit = render_inline_unit_input(col)
        # Now label_with_unit(col) will return "ColName (unit)"

    Parameters:
        col (str): Column name to define the unit for
        key_suffix (str): Extra string to make the widget key unique per page

    Returns:
        str: The entered unit string (empty string if nothing entered)
    """
    if "col_units" not in st.session_state:   # Initialize units dict if not yet created
        st.session_state["col_units"] = {}     # Create empty dict to store all units

    existing = st.session_state["col_units"].get(col, "")  # Get any previously entered unit

    unit = st.text_input(                      # Small text input for the unit
        f"Unit for {col} (optional):",         # Label telling user what to enter
        value=existing,                        # Pre-fill with previously entered unit
        placeholder="e.g. $, kg, years, %",   # Example hints
        key=f"unit_{col}_{key_suffix}"         # Unique key combining column name and suffix
    )

    st.session_state["col_units"][col] = unit.strip()  # Save unit to session state
    return unit.strip()                        # Return the unit so caller can use it directly
