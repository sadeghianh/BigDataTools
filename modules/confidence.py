# =========================
# modules/confidence.py
# Confidence Intervals module
# Computes a confidence interval for the mean of a numeric column.
# Written to be simple and clear for users with limited statistics background.
# Uses REAL data from the uploaded dataset.
# =========================

import pandas as pd               # Import pandas for DataFrame and Series operations
import numpy as np                # Import numpy for numerical operations
import scipy.stats as sp          # Import scipy.stats for the t and normal distributions
import streamlit as st            # Import streamlit for the user interface
from utils.helpers import (       # Import shared helper functions
    get_numeric_columns,          # Returns the list of numeric column names
    drop_missing,                 # Removes missing (NaN) values from a Series
    section_header,               # Shows a styled section heading
    label_with_unit,              # Builds an axis label that includes the unit
    render_inline_unit_input,     # Shows a small unit input next to a column selector
)


def render_confidence_intervals(df: pd.DataFrame):
    """
    Compute and display a confidence interval (CI) for the mean of a numeric column.

    A confidence interval gives a RANGE that very likely contains the true average
    of the whole population, based on our sample. For example, a 95% CI of
    [4,800 ; 5,200] means: "we are 95% confident the true average lies in this range."

    The module:
      - lets the user pick a numeric column and a confidence level (90%, 95%, 99%)
      - computes the sample mean, the margin of error, and the CI
      - shows the formula used and explains the result in plain language
      - draws a simple picture of the interval around the mean

    Parameters:
        df (pd.DataFrame): The uploaded dataset
    """
    section_header("Confidence Intervals", "🎯")  # Styled section heading

    # ---- Plain-language explanation shown to every user ----
    st.info("""
    **What is a Confidence Interval?**

    When you measure the average of a sample (e.g. the average salary of 200 employees),
    that average is just an estimate of the *true* average of everyone.
    A **confidence interval** gives a range around your estimate that very likely
    contains the true value.

    - A **95% confidence interval** means: if we repeated the study many times,
      about 95% of the intervals would contain the true average.
    - A **wider** interval means more uncertainty; a **narrower** one means more precision.
    """)  # Beginner-friendly explanation of the concept

    numeric_cols = get_numeric_columns(df)  # Get all numeric columns
    if not numeric_cols:  # If there are no numeric columns
        st.error("No numeric columns found in the dataset.")  # Show an error
        return  # Stop — nothing to compute

    # ---- User selects the column ----
    col = st.selectbox("Select a numeric column:", numeric_cols, key="ci_col")  # Column dropdown
    render_inline_unit_input(col, "ci")  # Optional unit input (e.g. $, kg) shown on results

    # ---- User selects the confidence level ----
    conf_level = st.selectbox(  # Dropdown for the confidence level
        "Confidence level:",
        ["95%", "90%", "99%"],  # Common confidence levels
        index=0,                # Default to 95%
        key="ci_level",
        help="How confident you want to be that the range contains the true mean. "
             "Higher confidence gives a wider range."
    )

    # ---- User chooses the method (t or z) ----
    method = st.radio(  # Radio buttons for the method
        "Method:",
        ["Automatic (recommended)", "t-distribution", "z-distribution (normal)"],
        horizontal=True,  # Lay options out in a row
        key="ci_method",
        help="t-distribution is best for small samples and when the population standard "
             "deviation is unknown. z is used for large samples. Automatic picks for you."
    )

    if st.button("Compute Confidence Interval"):  # Run only when the button is clicked
        series = drop_missing(df[col])  # Remove missing values from the column
        n = len(series)  # Sample size (number of observations)

        if n < 2:  # We need at least 2 values to measure spread
            st.error("Need at least 2 data points to compute a confidence interval.")  # Error
            return  # Stop

        # ---- Convert the confidence level text to a number ----
        conf_map = {"90%": 0.90, "95%": 0.95, "99%": 0.99}  # Map text to a fraction
        confidence = conf_map[conf_level]  # e.g. "95%" -> 0.95
        alpha = 1 - confidence  # The "tail" probability, e.g. 0.05 for 95%

        # ---- Compute the basic sample statistics ----
        mean = series.mean()  # Sample mean (our point estimate)
        std = series.std(ddof=1)  # Sample standard deviation (ddof=1 = sample, not population)
        se = std / np.sqrt(n)  # Standard error of the mean = std / sqrt(n)

        # ---- Decide whether to use the t or z distribution ----
        # t-distribution: best when n is small or population std is unknown (the usual case)
        # z-distribution: reasonable when n is large (>= 30)
        if method == "t-distribution":  # User forced t
            use_t = True  # Use t
        elif method == "z-distribution (normal)":  # User forced z
            use_t = False  # Use z
        else:  # Automatic: use t for small samples, z for large
            use_t = n < 30  # t if fewer than 30 observations, else z

        # ---- Compute the critical value and margin of error ----
        if use_t:  # Using the t-distribution
            # The critical t-value depends on the confidence level AND the sample size (df = n-1)
            df_t = n - 1  # Degrees of freedom for the t-distribution
            crit = sp.t.ppf(1 - alpha / 2, df_t)  # Two-tailed critical t-value
            dist_name = f"t-distribution (df = {df_t})"  # For display
        else:  # Using the z-distribution (standard normal)
            crit = sp.norm.ppf(1 - alpha / 2)  # Two-tailed critical z-value
            dist_name = "z-distribution (standard normal)"  # For display

        margin = crit * se  # Margin of error = critical value × standard error
        lower = mean - margin  # Lower bound of the interval
        upper = mean + margin  # Upper bound of the interval

        # ---- Build the unit suffix for display ----
        from utils.helpers import get_unit  # Import here to fetch the saved unit
        unit = get_unit(col)  # Get the unit the user typed (may be empty)
        u = f" {unit}" if unit else ""  # A space + unit, or nothing

        # ---- Show the main result as metrics ----
        st.markdown("---")  # Divider
        st.markdown(f"### Result: {conf_level} Confidence Interval for the mean of '{col}'")  # Heading

        c1, c2, c3 = st.columns(3)  # Three metric columns
        with c1:  # First metric
            st.metric("Sample Mean", f"{mean:.2f}{u}")  # The point estimate
        with c2:  # Second metric
            st.metric("Margin of Error", f"± {margin:.2f}{u}")  # The +/- amount
        with c3:  # Third metric
            st.metric("Sample Size (n)", str(n))  # Number of observations

        # ---- The interval itself, stated clearly ----
        st.success(
            f"🎯 **{conf_level} Confidence Interval: [ {lower:.2f}{u} , {upper:.2f}{u} ]**\n\n"
            f"We are {conf_level} confident that the true average of '{col}' for the whole "
            f"population lies between **{lower:.2f}{u}** and **{upper:.2f}{u}**."
        )  # Plain-language statement of the result

        # ---- Show the formula and the numbers that went into it ----
        with st.expander("📖 Formula and calculation details", expanded=True):  # Collapsible details
            st.latex(r"CI = \bar{x} \pm \left( \text{critical value} \times \frac{s}{\sqrt{n}} \right)")  # CI formula
            st.write("**Where:**")  # Label
            st.write(f"- x̄ (sample mean) = **{mean:.4f}{u}**")  # Mean value
            st.write(f"- s (sample standard deviation) = **{std:.4f}{u}**")  # Std value
            st.write(f"- n (sample size) = **{n}**")  # n value
            st.write(f"- Standard error (s / √n) = **{se:.4f}{u}**")  # SE value
            st.write(f"- Distribution used = **{dist_name}**")  # Which distribution
            st.write(f"- Critical value = **{crit:.4f}**")  # Critical value
            st.write(f"- Margin of error (critical × SE) = **{margin:.4f}{u}**")  # Margin
            st.write(f"- **Interval = {mean:.4f} ± {margin:.4f} = [{lower:.4f}, {upper:.4f}]{u}**")  # Final

        # ---- Draw a simple visual of the interval ----
        _plot_confidence_interval(mean, lower, upper, col, unit)  # Call the plotting helper

        # ---- Show how the interval changes with confidence level (teaching aid) ----
        with st.expander("📊 Compare confidence levels (90% vs 95% vs 99%)"):  # Collapsible comparison
            st.write("Notice how higher confidence gives a WIDER interval "
                     "(more certainty requires more room):")  # Explanation
            rows = []  # Collect rows for the comparison table
            for lvl_text, lvl in [("90%", 0.90), ("95%", 0.95), ("99%", 0.99)]:  # Each level
                a = 1 - lvl  # Tail probability
                if use_t:  # t-distribution
                    cv = sp.t.ppf(1 - a / 2, n - 1)  # Critical t
                else:  # z-distribution
                    cv = sp.norm.ppf(1 - a / 2)  # Critical z
                m = cv * se  # Margin for this level
                rows.append({  # Add a row
                    "Confidence Level": lvl_text,  # Level
                    "Lower Bound": round(mean - m, 2),  # Lower
                    "Upper Bound": round(mean + m, 2),  # Upper
                    "Width": round(2 * m, 2),  # Total width
                })  # End row
            comp_df = pd.DataFrame(rows)  # Build the comparison DataFrame
            st.dataframe(comp_df, use_container_width=True, hide_index=True)  # Show the table


def _plot_confidence_interval(mean, lower, upper, col, unit):
    """
    Draw a simple horizontal picture of the confidence interval:
    a dot for the mean and a bar showing the lower-to-upper range.

    Parameters:
        mean (float): The sample mean (center point)
        lower (float): Lower bound of the interval
        upper (float): Upper bound of the interval
        col (str): Column name (for the axis label)
        unit (str): Unit string (may be empty)
    """
    import matplotlib.pyplot as plt  # Import matplotlib for the plot

    fig, ax = plt.subplots(figsize=(9, 2.2))  # Create a short, wide figure

    # Draw the horizontal interval line (the range)
    ax.plot([lower, upper], [0, 0], color="#5B8FF9", linewidth=4, solid_capstyle="round",
            label="Confidence interval")  # The blue range bar

    # Draw vertical "caps" at the lower and upper bounds
    ax.plot([lower, lower], [-0.1, 0.1], color="#5B8FF9", linewidth=3)  # Left cap
    ax.plot([upper, upper], [-0.1, 0.1], color="#5B8FF9", linewidth=3)  # Right cap

    # Draw a red dot at the mean (the point estimate)
    ax.plot(mean, 0, "o", color="#E8684A", markersize=14, label="Sample mean", zorder=5)  # Mean dot

    # Annotate the three key numbers above the line
    u = f" {unit}" if unit else ""  # Unit suffix
    ax.annotate(f"{lower:.1f}{u}", (lower, 0.18), ha="center", fontsize=10, color="#333")  # Lower label
    ax.annotate(f"{upper:.1f}{u}", (upper, 0.18), ha="center", fontsize=10, color="#333")  # Upper label
    ax.annotate(f"mean = {mean:.1f}{u}", (mean, -0.28), ha="center", fontsize=10,
                fontweight="bold", color="#E8684A")  # Mean label

    ax.set_ylim(-0.5, 0.5)  # Fix the vertical range so the bar sits in the middle
    ax.set_yticks([])  # Hide the y-axis ticks (not meaningful here)
    ax.set_xlabel(label_with_unit(col))  # Label the x-axis with the column name and unit
    ax.set_title("Confidence Interval around the mean", fontsize=12, fontweight="bold")  # Title
    ax.legend(loc="upper right", fontsize=9)  # Show the legend
    ax.grid(axis="x", alpha=0.3)  # Faint vertical grid lines for reading values

    plt.tight_layout()  # Prevent labels from being cut off
    st.pyplot(fig)  # Render the figure in Streamlit
    plt.close(fig)  # Free memory
