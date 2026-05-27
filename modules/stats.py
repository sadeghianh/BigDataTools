# =========================
# modules/stats.py
# Descriptive statistics module
# =========================

# Import pandas for DataFrame and data manipulation
import pandas as pd  # Import pandas for DataFrame and data manipulation
# Import numpy for numerical array operations and math functions
import numpy as np  # Import numpy for numerical array operations and math functions
# Import scipy.stats for statistical tests and distributions
from scipy import stats as sp  # Import scipy.stats for statistical distributions and tests
# Import streamlit to build the interactive web UI
import streamlit as st  # Import streamlit to build the interactive web UI
# Import utils library
from utils.helpers import get_numeric_columns, drop_missing, section_header, label_with_unit, render_inline_unit_input  # Import utils.helpers library


# Define the render_statistics function
def render_statistics(df: pd.DataFrame):  # Define the render_statistics function
    # Execute this operation
    section_header("Descriptive Statistics", "📐")  # Execute this statement

    # Pandas data operation
    numeric_cols = get_numeric_columns(df)  # Store result in numeric_cols
    # Check condition and branch accordingly
    if not numeric_cols:  # Check condition
        # Show a red error message to the user
        st.error("No numeric columns found in the dataset.")  # Show a red error message
        # Exit the function early — stop execution here
        return  # Exit function early

    # Compute and store the result in col
    col = st.selectbox("Select a numeric column:", numeric_cols, key="stats_col")  # Store result in col

    # Ask user to enter the unit for the selected column
    # render_inline_unit_input saves unit to session_state and returns it
    unit = render_inline_unit_input(col, "stats")  # Show unit input and get the entered unit

    # Pandas data operation
    series = df[col]  # Store result in series
    # Pandas data operation
    series_clean = drop_missing(series)  # Store result in series_clean

    # Build display suffix — appended to every result value
    # Compute and store the result in unit_label
    unit_label = f" {unit.strip()}" if unit.strip() else ""  # Store result in unit_label

    # Show small grey caption text below a widget
    st.caption(f"Using {len(series_clean)} of {len(series)} values (after removing NaN)")  # Show small grey caption text

    # Compute and store the result in col1, col2, col3, col4, col5
    col1, col2, col3, col4, col5 = st.columns(5)  # Store result in col1, col2, col3, col4, col5

    # Open a context manager (auto-closes when done)
    with col1:  # Open context manager
        # Check condition and branch accordingly
        if st.button("📊 Mean"):  # Check condition
            # Pandas data operation
            _show_mean(series_clean, col, unit_label)  # Execute this statement

    # Open a context manager (auto-closes when done)
    with col2:  # Open context manager
        # Check condition and branch accordingly
        if st.button("📊 Median"):  # Check condition
            # Pandas data operation
            _show_median(series_clean, col, unit_label)  # Execute this statement

    # Open a context manager (auto-closes when done)
    with col3:  # Open context manager
        # Check condition and branch accordingly
        if st.button("📊 Mode"):  # Check condition
            # Pandas data operation
            _show_mode(series_clean, col, unit_label)  # Execute this statement

    # Open a context manager (auto-closes when done)
    with col4:  # Open context manager
        # Check condition and branch accordingly
        if st.button("📊 Variance"):  # Check condition
            # Pandas data operation
            _show_variance(series_clean, col, unit_label)  # Execute this statement

    # Open a context manager (auto-closes when done)
    with col5:  # Open context manager
        # Check condition and branch accordingly
        if st.button("📊 Std Dev"):  # Check condition
            # Pandas data operation
            _show_stddev(series_clean, col, unit_label)  # Execute this statement

    # Render formatted markdown text in the Streamlit UI
    st.markdown("---")  # Render formatted markdown text in the Streamlit UI
    # Render formatted markdown text in the Streamlit UI
    st.markdown("#### 📋 Full Summary Table")  # Render formatted markdown text in the Streamlit UI
    # Pandas data operation
    _show_summary_table(series_clean, col, unit_label)  # Execute this statement


# Define the _show_mean function
def _show_mean(series: pd.Series, col_name: str, unit: str = ""):  # Define the _show_mean function
    """Compute and display the arithmetic mean."""
    # Compute the arithmetic mean (average) of the values
    mean_val = np.mean(series)  # Compute arithmetic mean
    # Open a context manager (auto-closes when done)
    with st.expander(f"📊 Mean of '{col_name}'", expanded=True):  # Open context manager
        # Display a large highlighted metric value
        st.metric("Mean", f"{mean_val:.4f}{unit}")  # Display a large highlighted metric value
        # Render a mathematical formula using LaTeX notation
        st.latex(r"\bar{x} = \frac{\sum x_i}{n}")  # Render a LaTeX mathematical formula
        # Show small grey caption text below a widget
        st.caption(f"Sum = {np.sum(series):.4f}{unit}, n = {len(series)}")  # Show small grey caption text


# Define the _show_median function
def _show_median(series: pd.Series, col_name: str, unit: str = ""):  # Define the _show_median function
    """Compute and display the median."""
    # Compute the median (middle value) of the sorted data
    median_val = np.median(series)  # Compute median value
    # Open a context manager (auto-closes when done)
    with st.expander(f"📊 Median of '{col_name}'", expanded=True):  # Open context manager
        # Display a large highlighted metric value
        st.metric("Median", f"{median_val:.4f}{unit}")  # Display a large highlighted metric value
        # Render a mathematical formula using LaTeX notation
        st.latex(r"\text{Median} = \text{middle value of sorted data}")  # Render a LaTeX mathematical formula
        # Show small grey caption text below a widget
        st.caption("If n is even, the median is the average of the two middle values.")  # Show small grey caption text


# Define the _show_mode function
def _show_mode(series: pd.Series, col_name: str, unit: str = ""):  # Define the _show_mode function
    """Compute and display the mode."""
    # Find the most frequently occurring value in the series
    mode_result = sp.mode(series, keepdims=True)  # Find the most frequent value (mode)
    # Compute and store the result in mode_val
    mode_val = mode_result.mode[0]  # Store result in mode_val
    # Compute and store the result in mode_count
    mode_count = mode_result.count[0]  # Store result in mode_count
    # Open a context manager (auto-closes when done)
    with st.expander(f"📊 Mode of '{col_name}'", expanded=True):  # Open context manager
        # Display a large highlighted metric value
        st.metric("Mode", f"{mode_val:.4f}{unit}")  # Display a large highlighted metric value
        # Show small grey caption text below a widget
        st.caption(f"Appears {mode_count} times in the dataset.")  # Show small grey caption text
        # Render a mathematical formula using LaTeX notation
        st.latex(r"\text{Mode} = \text{most frequently occurring value}")  # Render a LaTeX mathematical formula


# Define the _show_variance function
def _show_variance(series: pd.Series, col_name: str, unit: str = ""):  # Define the _show_variance function
    """Compute and display sample variance. Variance unit = unit squared."""
    # Compute the variance (squared spread) of the values
    variance_val = np.var(series, ddof=1)  # Compute variance
    # Compute and store the result in unit_sq
    unit_sq = f" {unit.strip()}²" if unit.strip() else ""  # Store result in unit_sq
    # Open a context manager (auto-closes when done)
    with st.expander(f"📊 Variance of '{col_name}'", expanded=True):  # Open context manager
        # Display a large highlighted metric value
        st.metric("Variance", f"{variance_val:.4f}{unit_sq}")  # Display a large highlighted metric value
        # Render a mathematical formula using LaTeX notation
        st.latex(r"s^2 = \frac{\sum (x_i - \bar{x})^2}{n - 1}")  # Render a LaTeX mathematical formula
        # Show small grey caption text below a widget
        st.caption("ddof=1 means sample variance (divided by n−1, not n).")  # Show small grey caption text


# Define the _show_stddev function
def _show_stddev(series: pd.Series, col_name: str, unit: str = ""):  # Define the _show_stddev function
    """Compute and display sample standard deviation."""
    # Compute the standard deviation (spread) of the values
    std_val = np.std(series, ddof=1)  # Compute standard deviation
    # Open a context manager (auto-closes when done)
    with st.expander(f"📊 Std Deviation of '{col_name}'", expanded=True):  # Open context manager
        # Display a large highlighted metric value
        st.metric("Standard Deviation", f"{std_val:.4f}{unit}")  # Display a large highlighted metric value
        # Render a mathematical formula using LaTeX notation
        st.latex(r"s = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n - 1}}")  # Render a LaTeX mathematical formula
        # Show small grey caption text below a widget
        st.caption(f"Std dev is in the same units as the original variable{unit}.")  # Show small grey caption text


# Define the _show_summary_table function
def _show_summary_table(series: pd.Series, col_name: str, unit: str = ""):  # Define the _show_summary_table function
    """Display full summary table with optional Unit column."""
    # Compute and store the result in unit_sq
    unit_sq = f" {unit.strip()}²" if unit.strip() else ""  # Store result in unit_sq

    # Compute and store the result in stats_dict
    stats_dict = {  # Store result in stats_dict
        # Execute this operation
        "Statistic": [  # Execute this statement
            # Execute this operation
            "Count", "Mean", "Median", "Mode",  # Execute this statement
            # Execute this operation
            "Variance", "Std Dev", "Min", "Max",  # Execute this statement
            # Execute this operation
            "Range", "Skewness", "Kurtosis"  # Execute this statement
        # Execute this operation
        ],  # Execute this statement
        # Execute this operation
        "Value": [  # Execute this statement
            # Pandas data operation
            len(series),  # Execute this statement
            # Compute the arithmetic mean (average) of the values
            np.mean(series),  # Compute arithmetic mean
            # Compute the median (middle value) of the sorted data
            np.median(series),  # Compute median value
            # Find the most frequently occurring value in the series
            sp.mode(series, keepdims=True).mode[0],  # Find the most frequent value (mode)
            # Compute the variance (squared spread) of the values
            np.var(series, ddof=1),  # Compute variance
            # Compute the standard deviation (spread) of the values
            np.std(series, ddof=1),  # Compute standard deviation
            # NumPy numerical operation
            np.min(series),  # NumPy numerical computation
            # NumPy numerical operation
            np.max(series),  # NumPy numerical computation
            # NumPy numerical operation
            np.max(series) - np.min(series),  # NumPy numerical computation
            # Compute skewness — measure of distribution asymmetry
            sp.skew(series),  # Compute skewness of the distribution
            # Compute kurtosis — measure of distribution tail heaviness
            sp.kurtosis(series),  # Compute kurtosis of the distribution
        # Execute this operation
        ],  # Execute this statement
        # Execute this operation
        "Unit": [  # Execute this statement
            "",        # Count
            unit,      # Mean
            unit,      # Median
            unit,      # Mode
            unit_sq,   # Variance (unit²)
            unit,      # Std Dev
            unit,      # Min
            unit,      # Max
            unit,      # Range
            "",        # Skewness (dimensionless)
            "",        # Kurtosis (dimensionless)
        ]
    }

    # Create a new DataFrame from a dictionary or array
    summary_df = pd.DataFrame(stats_dict)  # Create DataFrame from dictionary or array
    # Apply a function to each element or row/column
    summary_df["Value"] = summary_df["Value"].apply(  # Store result in summary_df
        # Execute this operation
        lambda x: round(x, 4) if isinstance(x, float) else x  # Execute this statement
    )

    # Only show Unit column if user typed something
    # Check condition and branch accordingly
    if not unit.strip():  # Check condition
        # Remove specified columns or rows from the DataFrame
        summary_df = summary_df.drop(columns=["Unit"])  # Store result in summary_df

    # Render an interactive data table in the UI
    st.dataframe(summary_df, use_container_width=True, hide_index=True)  # Render an interactive data table
