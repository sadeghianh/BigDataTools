# =========================
# modules/sampling.py
# Sampling module — uses real uploaded dataset
# Implements Random, Systematic, and Stratified sampling
# with previews, formulas, and download options
#
# Fix: replaced groupby().apply() with manual loop to avoid pandas FutureWarning
# =========================

import pandas as pd               # Import pandas for DataFrame operations
import numpy as np                # Import numpy for array and index generation
import streamlit as st            # Import streamlit for UI rendering
from utils.helpers import (       # Import shared utility functions
    get_categorical_columns,      # To find categorical columns for stratification
    section_header,               # To display a styled section heading
)


def render_sampling(df: pd.DataFrame):  # Define the render_sampling function
    """
    Main function for the Sampling module.  # Execute this statement
    Shows dataset info, lets user choose a sampling method,  # Execute this statement
    and renders the appropriate parameters and results.  # Execute this statement

    Parameters:  # Execute this statement
        df (pd.DataFrame): The uploaded dataset  # Create DataFrame from dictionary or array
    """
    section_header("Sampling Methods", "🎲")  # Display styled section heading

    # Show dataset size so user can choose a reasonable sample size
    st.write(f"Dataset has **{len(df)} rows** and **{len(df.columns)} columns**.")  # Display text or data in the Streamlit UI

    # Dropdown to select which sampling method to use
    method = st.selectbox(  # Store result in method
        "Select sampling method:",  # Execute this statement
        ["Random Sampling", "Systematic Sampling", "Stratified Sampling"],  # Execute this statement
        key="sampling_method"    # Unique widget key
    )

    # ---- RANDOM SAMPLING ----
    if method == "Random Sampling":  # Check condition
        # Slider to choose how many rows to draw
        n = st.slider(  # Store result in n
            "Number of samples:",  # Execute this statement
            min_value=1,  # Store result in min_value
            max_value=len(df),          # Cannot sample more than total rows
            value=min(50, len(df)),     # Default = 50 or dataset size (whichever smaller)
            key="rand_n"  # Store result in key
        )
        if st.button("Draw Random Sample"):    # Only run when button is clicked
            _random_sampling(df, n)            # Call random sampling function

    # ---- SYSTEMATIC SAMPLING ----
    elif method == "Systematic Sampling":  # Check alternative condition
        # Slider to choose the step size k (take every k-th row)
        k = st.slider(  # Store result in k
            "Select every k-th row (step size):",  # Execute this statement
            min_value=2,  # Store result in min_value
            max_value=max(2, len(df) // 2),  # Maximum step = half the dataset
            value=5,                          # Default step = every 5th row
            key="sys_k"  # Store result in key
        )
        if st.button("Draw Systematic Sample"):    # Trigger button
            _systematic_sampling(df, k)            # Call systematic sampling function

    # ---- STRATIFIED SAMPLING ----
    elif method == "Stratified Sampling":  # Check alternative condition
        cat_cols = get_categorical_columns(df)  # Stratification needs a categorical column

        if not cat_cols:  # Check condition
            # Cannot do stratified sampling without a categorical column
            st.warning("Stratified sampling requires at least one categorical column in your dataset.")  # Show a yellow warning message
        else:
            # Let user choose which categorical column defines the strata (groups)
            strat_col = st.selectbox("Select stratification column:", cat_cols, key="strat_col")  # Store result in strat_col

            # Slider for the fraction to sample from each stratum
            frac = st.slider(  # Store result in frac
                "Fraction per stratum (0.0 to 1.0):",  # Execute this statement
                min_value=0.05,   # Minimum 5% per group
                max_value=1.0,    # Maximum 100% per group
                value=0.3,        # Default = 30% per group
                step=0.05,        # Change in increments of 5%
                key="strat_frac"  # Store result in key
            )
            if st.button("Draw Stratified Sample"):    # Trigger button
                _stratified_sampling(df, strat_col, frac)   # Call stratified sampling function


# =====================================================================
# Individual sampling functions
# =====================================================================

def _random_sampling(df: pd.DataFrame, n: int):  # Define the _random_sampling function
    """
    Simple Random Sampling — each row has equal probability of being selected.  # Execute this statement
    Uses sampling without replacement (no row appears twice).  # Execute this statement

    Parameters:  # Execute this statement
        df (pd.DataFrame): Full dataset  # Create DataFrame from dictionary or array
        n (int): Number of rows to sample  # Execute this statement
    """
    # df.sample(n) randomly picks n rows without replacement
    # random_state=42 makes the result reproducible (same rows every run)
    sample = df.sample(n=n, random_state=42)  # Store result in sample

    # Show results inside an expandable section
    with st.expander(f"✅ Random Sample — {n} rows", expanded=True):  # Open context manager
        st.dataframe(sample, use_container_width=True)   # Display the sample table

        # Show how much of the dataset was sampled
        st.caption(f"Sampled {n} rows from {len(df)} total ({100*n/len(df):.1f}%)")  # Show small grey caption text

        # Explain the method and show the formula used
        st.info(f"""  # Show an informational blue message box
        **Method:** Simple Random Sampling (without replacement)  # Execute this statement
        **Formula:** P(row i selected) = n / N = {n} / {len(df)} = {n/len(df):.4f}  # Store result in **Formula:** P(row i selected)
        **Implementation:** `df.sample(n={n}, random_state=42)`  # Store result in sample(n
        **Result:** {n} rows selected uniformly at random from {len(df)} total rows  # Execute this statement
        """)

        # Allow user to download the sample as a CSV file
        csv = sample.to_csv(index=False).encode("utf-8")   # Convert to bytes for download
        st.download_button("⬇️ Download Sample CSV", csv, "random_sample.csv", "text/csv")  # Create a file download button


def _systematic_sampling(df: pd.DataFrame, k: int):  # Define the _systematic_sampling function
    """
    Systematic Sampling — select every k-th row starting from a random position.  # Execute this statement
    Ensures even spacing across the dataset.  # Execute this statement
    Useful when data has a natural order (e.g. time series).  # Execute this statement

    Parameters:  # Execute this statement
        df (pd.DataFrame): Full dataset  # Create DataFrame from dictionary or array
        k (int): Step size — select every k-th row  # Execute this statement
    """
    np.random.seed(42)                    # Fix seed for reproducibility
    start = np.random.randint(0, k)       # Random starting index between 0 and k-1

    # Generate row indices: start, start+k, start+2k, ...
    # np.arange creates evenly spaced values from start to end of dataset
    indices = np.arange(start, len(df), k)  # Create integer array with even spacing

    # Select rows at the computed indices using .iloc (integer position indexing)
    sample = df.iloc[indices]  # Store result in sample

    with st.expander(f"✅ Systematic Sample — every {k}th row", expanded=True):  # Open context manager
        st.dataframe(sample, use_container_width=True)   # Display the sample

        # Show which rows were selected and summary
        st.caption(f"Starting at row {start}, selecting every {k}th row. Total: {len(sample)} rows.")  # Show small grey caption text

        # Explain the method with the actual parameters used
        st.info(f"""  # Show an informational blue message box
        **Method:** Systematic Sampling  # Execute this statement
        **Formula:** Selected indices = start, start+k, start+2k, ...  # Store result in **Formula:** Selected indices
        **Parameters used:** start = {start} (random), k = {k} (step size)  # Store result in **Parameters used:** start
        **Implementation:** `df.iloc[np.arange({start}, {len(df)}, {k})]`  # Create integer array with even spacing
        **Result:** {len(sample)} rows selected from {len(df)} total (every {k}th row)  # Execute this statement
        """)

        # Download button for the sample
        csv = sample.to_csv(index=False).encode("utf-8")  # Store result in csv
        st.download_button("⬇️ Download Sample CSV", csv, "systematic_sample.csv", "text/csv")  # Create a file download button


def _stratified_sampling(df: pd.DataFrame, strat_col: str, frac: float):  # Define the _stratified_sampling function
    """
    Stratified Sampling — sample a fixed proportion from each group (stratum).  # Execute this statement
    Guarantees every category in the stratification column is represented.  # Execute this statement

    FIX: Uses a manual for-loop instead of groupby().apply() to avoid  # Execute this statement
    the pandas FutureWarning about groupby behavior changes in newer versions.  # Execute this statement

    Parameters:  # Execute this statement
        df (pd.DataFrame): Full dataset  # Create DataFrame from dictionary or array
        strat_col (str): Column name used to define the groups (strata)  # Execute this statement
        frac (float): Fraction of each group to sample (e.g. 0.3 = 30%)  # Store result in 3
    """
    try:
        chunks = []       # Will hold the sampled rows from each group
        group_info = {}   # Will hold per-group statistics for display

        # Manual loop over each group — avoids groupby().apply() FutureWarning
        for group_name, group_df in df.groupby(strat_col):  # Group data by a column for aggregation
            n_group = len(group_df)   # Total rows in this group

            # Calculate how many rows to sample from this group
            # max(1, ...) ensures we always take at least 1 row
            # min(1.0, frac) ensures we never request more than 100%
            n_sample = max(1, int(np.floor(n_group * min(1.0, frac))))  # Round down to nearest integer

            # Sample from this group with fixed random state for reproducibility
            sampled = group_df.sample(n=n_sample, random_state=42)  # Store result in sampled

            chunks.append(sampled)   # Add this group's sample to the list

            # Store info about this group for the summary table
            group_info[str(group_name)] = {  # Store result in group_info
                "Total in group": n_group,                              # How many rows exist
                "Sampled": n_sample,                                    # How many were selected
                "Fraction": f"{100*n_sample/n_group:.1f}%"             # Actual % sampled
            }

        # Combine all group samples into one DataFrame
        # reset_index(drop=True) gives clean 0-based row numbers
        sample = pd.concat(chunks).reset_index(drop=True)  # Concatenate multiple DataFrames row-wise

        with st.expander(f"✅ Stratified Sample by '{strat_col}'", expanded=True):  # Open context manager
            st.dataframe(sample, use_container_width=True)   # Show the combined sample

            # Show per-group breakdown as a table
            st.markdown("**Sampling details per stratum:**")  # Render formatted markdown text in the Streamlit UI
            info_df = pd.DataFrame(group_info).T.reset_index()   # Transpose for readability
            info_df.columns = ["Group", "Total in group", "Sampled", "Fraction"]  # Rename columns
            st.dataframe(info_df, use_container_width=True, hide_index=True)  # Render an interactive data table

            # Show overall summary
            st.caption(  # Show small grey caption text
                f"Total sampled: {len(sample)} rows from {len(df)} total "  # Execute this statement
                f"({100*len(sample)/len(df):.1f}%)"  # Execute this statement
            )

            # Explain the method with formula and parameters used
            st.info(f"""  # Show an informational blue message box
            **Method:** Stratified Sampling  # Execute this statement
            **Formula:** n_i = floor(N_i × frac) for each stratum i  # Store result in **Formula:** n_i
            **Parameters used:** frac = {frac} ({int(frac*100)}% per group)  # Store result in **Parameters used:** frac
            **Implementation:** Manual loop sampling {int(frac*100)}% from each '{strat_col}' group  # Execute this statement
            **Result:** {len(sample)} rows total, proportionally from each group  # Execute this statement
            """)

            # Download button for the stratified sample
            csv = sample.to_csv(index=False).encode("utf-8")   # Convert to bytes
            st.download_button("⬇️ Download Sample CSV", csv, "stratified_sample.csv", "text/csv")  # Create a file download button

    except Exception as e:  # Handle any error from the try block
        # Catch unexpected errors and show a user-friendly message without crashing
        st.error(f"Stratified sampling failed: {e}")  # Show a red error message
