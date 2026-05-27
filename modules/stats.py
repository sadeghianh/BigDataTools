# =========================
# modules/stats.py
# Descriptive statistics module
# =========================

import pandas as pd
import numpy as np
from scipy import stats as sp
import streamlit as st
from utils.helpers import get_numeric_columns, drop_missing, section_header


def render_statistics(df: pd.DataFrame):
    section_header("Descriptive Statistics", "📐")

    numeric_cols = get_numeric_columns(df)
    if not numeric_cols:
        st.error("No numeric columns found in the dataset.")
        return

    col = st.selectbox("Select a numeric column:", numeric_cols, key="stats_col")

    # Unit input — user types the unit for the selected column (e.g. $, kg, m)
    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption(f"Selected column: **{col}**")
    with c2:
        unit = st.text_input("Unit (optional):", value="", placeholder="e.g. $, kg, m", key="stats_unit")

    series = df[col]
    series_clean = drop_missing(series)

    # Build display suffix — appended to every result value
    unit_label = f" {unit.strip()}" if unit.strip() else ""

    st.caption(f"Using {len(series_clean)} of {len(series)} values (after removing NaN)")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("📊 Mean"):
            _show_mean(series_clean, col, unit_label)

    with col2:
        if st.button("📊 Median"):
            _show_median(series_clean, col, unit_label)

    with col3:
        if st.button("📊 Mode"):
            _show_mode(series_clean, col, unit_label)

    with col4:
        if st.button("📊 Variance"):
            _show_variance(series_clean, col, unit_label)

    with col5:
        if st.button("📊 Std Dev"):
            _show_stddev(series_clean, col, unit_label)

    st.markdown("---")
    st.markdown("#### 📋 Full Summary Table")
    _show_summary_table(series_clean, col, unit_label)


def _show_mean(series: pd.Series, col_name: str, unit: str = ""):
    """Compute and display the arithmetic mean."""
    mean_val = np.mean(series)
    with st.expander(f"📊 Mean of '{col_name}'", expanded=True):
        st.metric("Mean", f"{mean_val:.4f}{unit}")
        st.latex(r"\bar{x} = \frac{\sum x_i}{n}")
        st.caption(f"Sum = {np.sum(series):.4f}{unit}, n = {len(series)}")


def _show_median(series: pd.Series, col_name: str, unit: str = ""):
    """Compute and display the median."""
    median_val = np.median(series)
    with st.expander(f"📊 Median of '{col_name}'", expanded=True):
        st.metric("Median", f"{median_val:.4f}{unit}")
        st.latex(r"\text{Median} = \text{middle value of sorted data}")
        st.caption("If n is even, the median is the average of the two middle values.")


def _show_mode(series: pd.Series, col_name: str, unit: str = ""):
    """Compute and display the mode."""
    mode_result = sp.mode(series, keepdims=True)
    mode_val = mode_result.mode[0]
    mode_count = mode_result.count[0]
    with st.expander(f"📊 Mode of '{col_name}'", expanded=True):
        st.metric("Mode", f"{mode_val:.4f}{unit}")
        st.caption(f"Appears {mode_count} times in the dataset.")
        st.latex(r"\text{Mode} = \text{most frequently occurring value}")


def _show_variance(series: pd.Series, col_name: str, unit: str = ""):
    """Compute and display sample variance. Variance unit = unit squared."""
    variance_val = np.var(series, ddof=1)
    unit_sq = f" {unit.strip()}²" if unit.strip() else ""
    with st.expander(f"📊 Variance of '{col_name}'", expanded=True):
        st.metric("Variance", f"{variance_val:.4f}{unit_sq}")
        st.latex(r"s^2 = \frac{\sum (x_i - \bar{x})^2}{n - 1}")
        st.caption("ddof=1 means sample variance (divided by n−1, not n).")


def _show_stddev(series: pd.Series, col_name: str, unit: str = ""):
    """Compute and display sample standard deviation."""
    std_val = np.std(series, ddof=1)
    with st.expander(f"📊 Std Deviation of '{col_name}'", expanded=True):
        st.metric("Standard Deviation", f"{std_val:.4f}{unit}")
        st.latex(r"s = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n - 1}}")
        st.caption(f"Std dev is in the same units as the original variable{unit}.")


def _show_summary_table(series: pd.Series, col_name: str, unit: str = ""):
    """Display full summary table with optional Unit column."""
    unit_sq = f" {unit.strip()}²" if unit.strip() else ""

    stats_dict = {
        "Statistic": [
            "Count", "Mean", "Median", "Mode",
            "Variance", "Std Dev", "Min", "Max",
            "Range", "Skewness", "Kurtosis"
        ],
        "Value": [
            len(series),
            np.mean(series),
            np.median(series),
            sp.mode(series, keepdims=True).mode[0],
            np.var(series, ddof=1),
            np.std(series, ddof=1),
            np.min(series),
            np.max(series),
            np.max(series) - np.min(series),
            sp.skew(series),
            sp.kurtosis(series),
        ],
        "Unit": [
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

    summary_df = pd.DataFrame(stats_dict)
    summary_df["Value"] = summary_df["Value"].apply(
        lambda x: round(x, 4) if isinstance(x, float) else x
    )

    # Only show Unit column if user typed something
    if not unit.strip():
        summary_df = summary_df.drop(columns=["Unit"])

    st.dataframe(summary_df, use_container_width=True, hide_index=True)
