# =========================
# modules/data_profile.py
# Data Profile module — gives an overview of the uploaded dataset
# Shows: how many rows/columns, which are numeric/categorical,
# missing values, and for each numeric column whether it is normal.
# This helps the user understand their data before choosing a test.
# =========================

import pandas as pd               # Import pandas for DataFrame operations
import numpy as np                # Import numpy for numerical operations
import scipy.stats as sp          # Import scipy.stats for the normality test
import streamlit as st            # Import streamlit for UI rendering
from utils.helpers import (       # Import shared utility functions
    get_numeric_columns,          # Returns numeric column names
    get_categorical_columns,      # Returns categorical column names
    drop_missing,                 # Removes NaN values
    format_pvalue,                # Formats p-values
    section_header,               # Styled section header
)


def render_data_profile(df: pd.DataFrame):
    """
    Display a complete profile of the uploaded dataset.
    This is the "know your data" screen that answers:
      - How many observations (rows) do I have?
      - Which columns are numeric vs categorical?
      - How much data is missing?
      - Is each numeric column normally distributed?

    Knowing these facts tells the user which statistical tests are appropriate.

    Parameters:
        df (pd.DataFrame): The uploaded dataset
    """
    section_header("Data Profile — Know Your Data", "🔎")  # Styled heading

    numeric_cols = get_numeric_columns(df)  # Get numeric columns
    cat_cols = get_categorical_columns(df)  # Get categorical columns

    # ---- Top-level summary cards ----
    st.markdown("#### Dataset at a Glance")  # Section heading
    c1, c2, c3, c4 = st.columns(4)  # Four metric columns
    with c1:  # First card
        st.metric("Rows (observations)", f"{df.shape[0]:,}")  # Number of rows
    with c2:  # Second card
        st.metric("Columns (variables)", str(df.shape[1]))  # Number of columns
    with c3:  # Third card
        st.metric("Numeric columns", str(len(numeric_cols)))  # Count of numeric
    with c4:  # Fourth card
        st.metric("Categorical columns", str(len(cat_cols)))  # Count of categorical

    # ---- Sample size guidance ----
    n = df.shape[0]  # Total rows
    if n < 30:  # Small sample
        st.warning(f"⚠️ Your sample size is small (n = {n} < 30). "  # Show a yellow warning message
                   f"Non-parametric tests are recommended, and results should be interpreted cautiously.")  # Warn
    else:  # Adequate sample
        st.success(f"✅ Your sample size is adequate (n = {n} ≥ 30). "  # Show a green success message
                   f"The Central Limit Theorem supports the use of parametric tests.")  # Reassure

    st.markdown("---")  # Divider

    # ---- Missing values overview ----
    st.markdown("#### Missing Values")  # Section heading
    total_cells = df.shape[0] * df.shape[1]  # Total number of cells
    total_missing = int(df.isnull().sum().sum())  # Total missing cells
    missing_pct = 100 * total_missing / total_cells if total_cells > 0 else 0  # Percentage missing

    if total_missing == 0:  # No missing data
        st.success("✅ No missing values — your dataset is complete.")  # Good news
    else:  # Some missing data
        st.info(f"Total missing values: {total_missing:,} ({missing_pct:.1f}% of all cells)")  # Report
        # Build a table of columns that have missing values
        missing_by_col = df.isnull().sum()  # Count missing per column
        missing_by_col = missing_by_col[missing_by_col > 0]  # Keep only columns with missing
        if len(missing_by_col) > 0:  # If any
            miss_df = pd.DataFrame({  # Build table
                "Column": missing_by_col.index,  # Column names
                "Missing Count": missing_by_col.values,  # Missing counts
                "Missing %": (100 * missing_by_col.values / len(df)).round(1),  # Percentages
            })  # End DataFrame
            st.dataframe(miss_df, use_container_width=True, hide_index=True)  # Show table

    st.markdown("---")  # Divider

    # ---- Per-column type and normality table ----
    st.markdown("#### Column-by-Column Profile")  # Section heading
    st.caption("For each numeric column, we run a Shapiro-Wilk normality test "  # Show small grey caption text
               "to tell you whether parametric tests are appropriate.")  # Explain

    profile_rows = []  # List to collect rows for the profile table

    for col in df.columns:  # Loop over every column
        if col in numeric_cols:  # Numeric column
            series = drop_missing(df[col])  # Clean data
            n_col = len(series)  # Non-missing count
            # Run normality test if enough data
            if n_col >= 3:  # Enough to test
                test_data = series if n_col <= 5000 else series.sample(5000, random_state=42)  # Sample if huge
                try:
                    _, sw_p = sp.shapiro(test_data)  # Shapiro-Wilk p-value
                    if sw_p > 0.05:  # Normal
                        normality = "Normal ✓"  # Label
                    else:  # Not normal
                        normality = "Not normal"  # Label
                    norm_detail = f"p={format_pvalue(sw_p)}"  # p-value detail
                except Exception:  # Test failed
                    normality = "—"  # Unknown
                    norm_detail = ""  # No detail
            else:  # Too few points
                normality = "—"  # Cannot test
                norm_detail = "n<3"  # Reason
            profile_rows.append({  # Add a row for this numeric column
                "Column": col,  # Name
                "Type": "Numeric",  # Type
                "Non-Null": n_col,  # Count
                "Unique": series.nunique(),  # Unique values
                "Normality": normality,  # Normality result
                "Detail": norm_detail,  # p-value
            })
        elif col in cat_cols:  # Categorical column
            series = df[col].dropna()  # Clean data
            profile_rows.append({  # Add a row for this categorical column
                "Column": col,  # Name
                "Type": "Categorical",  # Type
                "Non-Null": len(series),  # Count
                "Unique": series.nunique(),  # Number of categories
                "Normality": "N/A (categorical)",  # Normality not applicable
                "Detail": f"{series.nunique()} categories",  # Category count
            })

    profile_df = pd.DataFrame(profile_rows)  # Convert collected rows to DataFrame
    st.dataframe(profile_df, use_container_width=True, hide_index=True)  # Show the profile table

    # ---- Summary recommendation ----
    st.markdown("---")  # Divider
    st.markdown("#### What This Means for Your Analysis")  # Section heading

    # Count how many numeric columns are normal vs not
    normal_count = sum(1 for r in profile_rows  # Count normal numeric columns
                       if r["Type"] == "Numeric" and "Normal ✓" in r["Normality"])
    numeric_count = sum(1 for r in profile_rows if r["Type"] == "Numeric")  # Total numeric

    if numeric_count > 0:  # If there are numeric columns
        st.write(f"- **{normal_count} of {numeric_count}** numeric columns appear normally distributed.")  # Report
        if n >= 30:  # Large sample
            st.write("- Because your sample is large (n ≥ 30), you can generally use **parametric tests** "  # Display explanatory text in the UI
                     "(t-test, ANOVA, Pearson) even for non-normal columns, thanks to the Central Limit Theorem.")  # Guidance
        else:  # Small sample
            st.write("- Because your sample is small (n < 30), prefer **parametric tests only for normal columns**. "  # Display explanatory text in the UI
                     "For non-normal columns, use **non-parametric tests** (Mann-Whitney, Kruskal-Wallis, Wilcoxon).")  # Guidance

    st.info("💡 Tip: Go to the **Test Advisor** (under Hypothesis Testing) to get a specific "  # Show an informational message
            "test recommendation based on your exact variables and goal.")  # Point to advisor

    # =====================================================================
    # DATA CLEANING & EXPORT
    # Lets the user clean the dataset (handle missing values, remove duplicates)
    # and download the cleaned result as a CSV file.
    # =====================================================================
    st.markdown("---")  # Divider
    st.markdown("#### 🧹 Clean & Export Data")  # Section heading
    st.caption("Optionally clean your dataset and download the result as a CSV file.")  # Explanation

    # ---- Let the user choose how to handle missing values ----
    clean_option = st.radio(  # Radio buttons for the cleaning strategy
        "How should missing values be handled?",
        [
            "Keep as-is (no change)",                 # Do nothing
            "Remove rows with any missing value",     # Drop incomplete rows
            "Fill numeric missing with column mean",  # Impute numeric columns
            "Fill numeric missing with column median",# Impute with median
        ],
        key="clean_missing_option"  # Unique widget key
    )

    # ---- Option to remove duplicate rows ----
    remove_dupes = st.checkbox("Also remove duplicate rows", value=False, key="clean_dupes")  # Checkbox

    # ---- Build the cleaned DataFrame based on the chosen options ----
    cleaned = df.copy()  # Start from a copy so the original is never changed
    numeric_cols_all = cleaned.select_dtypes(include=[np.number]).columns  # Numeric columns

    if clean_option == "Remove rows with any missing value":  # Drop rows with NaN
        cleaned = cleaned.dropna()  # Remove any row containing a missing value
    elif clean_option == "Fill numeric missing with column mean":  # Mean imputation
        for c in numeric_cols_all:  # Loop over numeric columns
            cleaned[c] = cleaned[c].fillna(cleaned[c].mean())  # Fill NaN with the column mean
    elif clean_option == "Fill numeric missing with column median":  # Median imputation
        for c in numeric_cols_all:  # Loop over numeric columns
            cleaned[c] = cleaned[c].fillna(cleaned[c].median())  # Fill NaN with the column median
    # (If "Keep as-is" is chosen, we make no changes)

    if remove_dupes:  # If the user wants duplicates removed
        before = len(cleaned)  # Row count before
        cleaned = cleaned.drop_duplicates()  # Remove exact duplicate rows
        removed = before - len(cleaned)  # How many were removed
        if removed > 0:  # If any duplicates were found
            st.caption(f"Removed {removed} duplicate row(s).")  # Report how many

    # ---- Show a short before/after summary so the user sees the effect ----
    cc1, cc2 = st.columns(2)  # Two columns for before/after
    with cc1:  # Before
        st.metric("Original rows", f"{len(df):,}")  # Original row count
    with cc2:  # After
        st.metric("Rows after cleaning", f"{len(cleaned):,}")  # Cleaned row count

    # ---- Download button for the cleaned dataset ----
    csv_bytes = cleaned.to_csv(index=False).encode("utf-8")  # Convert the cleaned data to CSV bytes
    st.download_button(  # Create the download button
        "⬇️ Download Cleaned Dataset (CSV)",  # Button label
        data=csv_bytes,                        # The CSV content
        file_name="cleaned_dataset.csv",       # Suggested filename
        mime="text/csv",                       # File type
        help="Download your dataset after applying the cleaning options above."  # Tooltip
    )
