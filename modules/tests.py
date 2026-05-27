# =========================
# modules/tests.py
# Hypothesis Testing module
# Implements: One-Sample T-test, Two-Sample T-test,
# Z-test, One-Way ANOVA, Two-Way ANOVA, Chi-Square test
#
# Fixes applied:
#   1. Two-sample T-test: added Option B — threshold split for numeric-only datasets
#   2. One-way ANOVA: added Tukey HSD post-hoc test to show WHICH groups differ
#   3. Z-test: default mu0 = sample mean instead of 0 (more meaningful starting point)
# =========================

import pandas as pd               # Import pandas for DataFrame and Series operations
import numpy as np                # Import numpy for numerical computations
import scipy.stats as sp          # Import scipy.stats for all statistical tests
import streamlit as st            # Import streamlit for UI rendering
import statsmodels.api as sm      # Import statsmodels for two-way ANOVA table
from statsmodels.formula.api import ols           # OLS regression used inside two-way ANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd  # Tukey HSD post-hoc test
from utils.helpers import (       # Import shared utility functions
    get_numeric_columns,          # Returns list of numeric column names
    get_categorical_columns,      # Returns list of categorical column names
    drop_missing,                 # Removes NaN values from a Series
    section_header,               # Displays a styled section title in the UI
    format_pvalue,                # Formats p-value for clean display
)


def render_tests(df: pd.DataFrame):  # Define the render_tests function
    """
    Main entry point for the Hypothesis Testing module.  # Execute this statement
    Shows a dropdown to select which test to run, then routes to the correct function.  # Execute this statement

    Parameters:  # Execute this statement
        df (pd.DataFrame): The uploaded dataset  # Create DataFrame from dictionary or array
    """
    section_header("Hypothesis Testing", "🧪")  # Display styled section heading

    # Dropdown to choose which statistical test to perform
    test_type = st.selectbox(  # Store result in test_type
        "Select hypothesis test:",  # Label shown above the dropdown
        [                               # All available tests listed here
            "One-Sample T-test",        # Compare one sample mean to a hypothesized value
            "Two-Sample T-test",        # Compare means of two independent groups
            "Z-test (One-Sample)",      # Large-sample mean comparison using normal distribution
            "One-Way ANOVA",            # Compare means across 3+ groups
            "Two-Way ANOVA",            # Test two factors and their interaction
            "Chi-Square Test",          # Test association between two categorical variables
            "Pearson Correlation",      # Measure linear relationship between two numeric variables
        ],
        key="test_type_select"          # Unique widget key to avoid Streamlit conflicts
    )

    # Route to the matching function based on user selection
    if test_type == "One-Sample T-test":  # Check condition
        _one_sample_ttest(df)           # Call one-sample T-test function
    elif test_type == "Two-Sample T-test":  # Check alternative condition
        _two_sample_ttest(df)           # Call two-sample T-test function
    elif test_type == "Z-test (One-Sample)":  # Check alternative condition
        _ztest_one_sample(df)           # Call Z-test function
    elif test_type == "One-Way ANOVA":  # Check alternative condition
        _one_way_anova(df)              # Call one-way ANOVA function
    elif test_type == "Two-Way ANOVA":  # Check alternative condition
        _two_way_anova(df)              # Call two-way ANOVA function
    elif test_type == "Chi-Square Test":  # Check alternative condition
        _chi_square_test(df)            # Call chi-square test function
    elif test_type == "Pearson Correlation":  # Check alternative condition
        _pearson_correlation(df)        # Call Pearson correlation function


# =====================================================================
# Individual test functions
# =====================================================================

def _one_sample_ttest(df: pd.DataFrame):  # Define the _one_sample_ttest function
    """
    One-sample T-test.  # Execute this statement
    Compares the mean of one sample to a hypothesized population mean (μ₀).  # Execute this statement

    H₀: μ = μ₀   (sample mean equals the hypothesized value)  # Store result in H₀: μ
    H₁: μ ≠ μ₀   (sample mean differs from the hypothesized value)  # Execute this statement

    FIX: Default μ₀ is now the sample mean (not 0) so user sees a real starting point.  # Execute this statement
    """
    st.markdown("### One-Sample T-test")                    # Sub-heading
    st.write("Tests whether the sample mean equals a hypothesized population mean.")  # Display text or data in the Streamlit UI

    numeric_cols = get_numeric_columns(df)   # Get all numeric columns from dataset
    if not numeric_cols:  # Check condition
        st.error("No numeric columns available.")  # Stop if no numeric data
        return  # Exit function early

    # Let user select which column to test
    col = st.selectbox("Select variable:", numeric_cols, key="t1_col")  # Store result in col

    # Extract column and remove NaN values
    series = drop_missing(df[col])  # Store result in series

    # FIX: Compute actual sample mean to use as the default μ₀
    # This is more meaningful than defaulting to 0, which may be irrelevant
    sample_mean = float(round(series.mean(), 4))  # Round to 4 decimal places for display

    # Number input for hypothesized mean — default = sample mean
    mu0 = st.number_input(  # Store result in mu0
        "Hypothesized mean (μ₀):",  # Execute this statement
        value=sample_mean,                                     # Default = actual sample mean
        step=float(max(0.01, sample_mean * 0.01)),             # Step size = 1% of mean
        key="t1_mu0",  # Store result in key
        help=f"Default = sample mean ({sample_mean}). Change to test a specific hypothesis."  # Store result in help
    )
    # Remind the user what the sample mean is and why they should change μ₀
    st.caption(f"Sample mean = {sample_mean} — change μ₀ to test against a different value (e.g. industry average).")  # Show small grey caption text

    # Dropdown for significance level α
    alpha = st.selectbox("Significance level (α):", [0.05, 0.01, 0.10], key="t1_alpha")  # Store result in alpha

    if st.button("Run One-Sample T-test"):              # Only run when user clicks button
        if len(series) < 2:  # Check condition
            st.error("Need at least 2 data points.")    # T-test requires at least 2 values
            return  # Exit function early

        # sp.ttest_1samp computes the t-statistic and p-value
        # popmean=mu0 is the value we're testing against
        t_stat, p_value = sp.ttest_1samp(series, popmean=mu0)  # One-sample T-test: compare mean to hypothesized value

        df_val = len(series) - 1   # Degrees of freedom = n - 1 for one-sample t-test

        # Display the results using the shared display function
        _display_test_results(  # Execute this statement
            test_name="One-Sample T-test",  # Store result in test_name
            statistic_name="t-statistic",  # Store result in statistic_name
            statistic=t_stat,  # Store result in statistic
            p_value=p_value,  # Store result in p_value
            df=df_val,  # Store result in df
            alpha=alpha,  # Store result in alpha
            h0=f"H₀: mean of '{col}' = {mu0}",           # Null hypothesis text
            h1=f"H₁: mean of '{col}' ≠ {mu0}",           # Alternative hypothesis text
            extra_info={  # Store result in extra_info
                "Sample Mean (x̄)": round(series.mean(), 4),           # Actual mean from data
                "Hypothesized Mean (μ₀)": mu0,                         # The value being tested
                "Difference (x̄ - μ₀)": round(series.mean() - mu0, 4), # How far off it is
                "Sample Std Dev (s)": round(series.std(ddof=1), 4),    # Spread of data
                "Sample Size (n)": len(series),                         # Number of observations
            }
        )

        # Check normality assumption — t-test assumes normal distribution
        _check_normality(series, col)  # Execute this statement


def _two_sample_ttest(df: pd.DataFrame):  # Define the _two_sample_ttest function
    """
    Two-sample independent T-test.  # Execute this statement
    Compares the means of two independent groups.  # Execute this statement

    H₀: μ₁ = μ₂   (two group means are equal)  # Store result in H₀: μ₁
    H₁: μ₁ ≠ μ₂   (two group means differ)  # Execute this statement

    FIX: Added Option B — split by numeric threshold when no categorical column exists.  # Execute this statement
    Levene's test automatically determines whether to use Student or Welch t-test.  # Execute this statement
    """
    st.markdown("### Two-Sample T-test")  # Render formatted markdown text in the Streamlit UI
    st.write("Tests whether the means of two independent groups are equal.")  # Display text or data in the Streamlit UI

    numeric_cols = get_numeric_columns(df)      # All numeric columns
    cat_cols = get_categorical_columns(df)       # All categorical columns

    if not numeric_cols:  # Check condition
        st.error("No numeric columns available.")  # Show a red error message
        return  # Exit function early

    # Decide which splitting method to show the user
    if cat_cols:  # Check condition
        # If categorical columns exist, offer both options
        split_method = st.radio(  # Store result in split_method
            "How to define the two groups?",  # Execute this statement
            ["Option A: Split by categorical column", "Option B: Split numeric column by threshold"],  # Execute this statement
            key="t2_split_method"  # Store result in key
        )
    else:
        # No categorical columns — only Option B is possible
        st.info("ℹ️ No categorical columns found. Using numeric threshold split (Option B).")  # Show an informational blue message box
        split_method = "Option B: Split numeric column by threshold"  # Force Option B

    # Let user choose which numeric column's mean to compare between the two groups
    num_col = st.selectbox("Numeric column to test:", numeric_cols, key="t2_num")  # Store result in num_col

    # ---- OPTION A: group by categorical column ----
    if split_method == "Option A: Split by categorical column":  # Check condition
        group_col = st.selectbox("Group column:", cat_cols, key="t2_grp")   # Categorical split column
        unique_groups = df[group_col].dropna().unique().tolist()             # All unique group values

        if len(unique_groups) < 2:  # Check condition
            st.warning("Need at least 2 groups in the selected column.")  # Show a yellow warning message
            return  # Exit function early

        # Let user pick exactly which two groups to compare
        g1_label = st.selectbox("Group 1:", unique_groups, key="t2_g1")  # Store result in g1_label
        g2_label = st.selectbox("Group 2:", [g for g in unique_groups if g != g1_label], key="t2_g2")  # Store result in g2_label
        alpha = st.selectbox("Significance level (α):", [0.05, 0.01, 0.10], key="t2_alpha_a")  # Store result in alpha

        if st.button("Run Two-Sample T-test"):  # Check condition
            # Filter rows for each group
            grp1 = drop_missing(df[df[group_col] == g1_label][num_col])  # Store result in grp1
            grp2 = drop_missing(df[df[group_col] == g2_label][num_col])  # Store result in grp2

            if len(grp1) < 2 or len(grp2) < 2:  # Check condition
                st.error("Each group needs at least 2 values.")  # Show a red error message
                return  # Exit function early

            # Levene's test checks if variances are equal
            # If p > 0.05: variances are equal → use Student t-test
            # If p ≤ 0.05: variances unequal → use Welch t-test
            _, levene_p = sp.levene(grp1, grp2)  # Levene's test for equal variance between groups
            equal_var = levene_p > 0.05  # True = equal variance assumed

            # sp.ttest_ind performs two-sample t-test
            # equal_var=True → Student's t, equal_var=False → Welch's t
            t_stat, p_value = sp.ttest_ind(grp1, grp2, equal_var=equal_var)  # Two-sample T-test: compare means of two groups
            df_val = len(grp1) + len(grp2) - 2   # Degrees of freedom = n1 + n2 - 2

            _display_test_results(  # Execute this statement
                test_name="Two-Sample T-test",  # Store result in test_name
                statistic_name="t-statistic",  # Store result in statistic_name
                statistic=t_stat,  # Store result in statistic
                p_value=p_value,  # Store result in p_value
                df=df_val,  # Store result in df
                alpha=alpha,  # Store result in alpha
                h0=f"H₀: mean({g1_label}) = mean({g2_label}) for '{num_col}'",  # Store result in h0
                h1=f"H₁: mean({g1_label}) ≠ mean({g2_label}) for '{num_col}'",  # Store result in h1
                extra_info={  # Store result in extra_info
                    f"Mean — {g1_label}": round(grp1.mean(), 4),  # Execute this statement
                    f"Mean — {g2_label}": round(grp2.mean(), 4),  # Execute this statement
                    f"n₁ ({g1_label})": len(grp1),  # Execute this statement
                    f"n₂ ({g2_label})": len(grp2),  # Execute this statement
                    "Levene test result": f"p={levene_p:.4f} → {'Equal var (Student t)' if equal_var else 'Unequal var (Welch t)'}",  # Store result in "Levene test result": f"p
                }
            )
            _check_normality(grp1, g1_label)   # Check normality for group 1
            _check_normality(grp2, g2_label)   # Check normality for group 2

    # ---- OPTION B: split numeric column by threshold ----
    else:
        series = drop_missing(df[num_col])   # Clean the selected column

        # Explain what Option B does
        st.info(f"""  # Show an informational blue message box
        **Option B — Threshold split:**  # Execute this statement
        Rows where '{num_col}' > threshold → **Group 1 (High)**  # Execute this statement
        Rows where '{num_col}' ≤ threshold → **Group 2 (Low)**  # Execute this statement
        """)

        # Default threshold = median — natural midpoint that creates roughly equal groups
        default_thresh = float(round(series.median(), 2))  # Store result in default_thresh
        threshold = st.number_input(  # Store result in threshold
            f"Threshold for '{num_col}':",  # Execute this statement
            value=default_thresh,                              # Start at the median
            step=float(max(0.01, series.std() * 0.1)),        # Step = 10% of std dev
            key="t2_threshold",  # Store result in key
            help=f"Default = median ({default_thresh}). Adjust to test different splits."  # Store result in help
        )
        st.caption(f"Median = {default_thresh} | Mean = {series.mean():.2f}")  # Show context

        # Choose which column to compare between the two groups
        compare_options = [c for c in numeric_cols if c != num_col]  # Exclude the split column
        if not compare_options:  # Check condition
            compare_options = numeric_cols    # Fallback if only one numeric column

        compare_col = st.selectbox(  # Store result in compare_col
            "Numeric column to compare between groups:", compare_options, key="t2_compare_col",  # Store result in "Numeric column to compare between groups:", compare_options, key
            help="The variable whose mean you want to compare across the two groups."  # Store result in help
        )
        alpha = st.selectbox("Significance level (α):", [0.05, 0.01, 0.10], key="t2_alpha_b")  # Store result in alpha

        if st.button("Run Two-Sample T-test"):  # Check condition
            # Create boolean mask to split data into high and low groups
            mask_high = df[num_col] > threshold       # True for Group 1 (above threshold)
            grp1 = drop_missing(df.loc[mask_high, compare_col])   # Group 1 values
            grp2 = drop_missing(df.loc[~mask_high, compare_col])  # Group 2 values (~mask = below)

            if len(grp1) < 2 or len(grp2) < 2:  # Check condition
                st.error(f"Each group needs ≥2 values. Got: Group1={len(grp1)}, Group2={len(grp2)}. Adjust threshold.")  # Show a red error message
                return  # Exit function early

            # Levene's test for variance equality
            _, levene_p = sp.levene(grp1, grp2)  # Levene's test for equal variance between groups
            equal_var = levene_p > 0.05  # Equal variance assumption

            t_stat, p_value = sp.ttest_ind(grp1, grp2, equal_var=equal_var)  # Run t-test
            df_val = len(grp1) + len(grp2) - 2   # Degrees of freedom

            _display_test_results(  # Execute this statement
                test_name="Two-Sample T-test (threshold split)",  # Store result in test_name
                statistic_name="t-statistic",  # Store result in statistic_name
                statistic=t_stat,  # Store result in statistic
                p_value=p_value,  # Store result in p_value
                df=df_val,  # Store result in df
                alpha=alpha,  # Store result in alpha
                h0=f"H₀: mean({num_col}>{threshold}) = mean({num_col}≤{threshold}) for '{compare_col}'",  # Store result in h0
                h1=f"H₁: mean({num_col}>{threshold}) ≠ mean({num_col}≤{threshold}) for '{compare_col}'",  # Store result in h1
                extra_info={  # Store result in extra_info
                    f"Group 1 ({num_col} > {threshold}) mean": round(grp1.mean(), 4),  # Execute this statement
                    f"Group 2 ({num_col} ≤ {threshold}) mean": round(grp2.mean(), 4),  # Execute this statement
                    "n₁ (High group)": len(grp1),  # Execute this statement
                    "n₂ (Low group)": len(grp2),  # Execute this statement
                    "Levene test result": f"p={levene_p:.4f} → {'Equal var (Student t)' if equal_var else 'Unequal var (Welch t)'}",  # Store result in "Levene test result": f"p
                }
            )


def _ztest_one_sample(df: pd.DataFrame):  # Define the _ztest_one_sample function
    """
    One-sample Z-test.  # Execute this statement
    Similar to T-test but uses normal distribution. Best for n ≥ 30.  # Execute this statement

    H₀: μ = μ₀    H₁: μ ≠ μ₀  # Store result in H₀: μ

    FIX: Default μ₀ = sample mean (not 0) so user starts from a meaningful value.  # Store result in FIX: Default μ₀
    Formula: z = (x̄ - μ₀) / (σ / √n)  # Store result in Formula: z
    """
    st.markdown("### Z-test (One-Sample)")  # Render formatted markdown text in the Streamlit UI
    st.write("Tests whether the sample mean equals a hypothesized value (best for n ≥ 30).")  # Display text or data in the Streamlit UI

    numeric_cols = get_numeric_columns(df)   # Get numeric columns
    if not numeric_cols:  # Check condition
        st.error("No numeric columns available.")  # Show a red error message
        return  # Exit function early

    col = st.selectbox("Select variable:", numeric_cols, key="z_col")  # Column dropdown
    series = drop_missing(df[col])   # Remove NaN values

    # FIX: Default μ₀ = sample mean so user starts from real data, not arbitrary 0
    sample_mean = float(round(series.mean(), 4))  # Compute actual mean for default
    mu0 = st.number_input(  # Store result in mu0
        "Hypothesized mean (μ₀):",  # Execute this statement
        value=sample_mean,                                     # Default = actual sample mean
        step=float(max(0.01, sample_mean * 0.01)),             # Step = 1% of mean
        key="z_mu0",  # Store result in key
        help=f"Default = sample mean ({sample_mean}). Change to test a specific benchmark."  # Store result in help
    )
    # Reminder to user to change the default
    st.caption(f"ℹ️ Sample mean = **{sample_mean}**. Change μ₀ to test a real hypothesis (e.g. industry benchmark).")  # Show small grey caption text

    # Option to use a known population σ instead of the sample std
    use_known_sigma = st.checkbox("Use known population σ?", value=False, key="z_known_sig")  # Store result in use_known_sigma
    if use_known_sigma:  # Check condition
        # If user knows σ from literature/prior studies, they can enter it here
        sigma = st.number_input(  # Store result in sigma
            "Population σ:",  # Execute this statement
            value=float(round(series.std(ddof=1), 4)),  # Default = sample std as starting point
            min_value=0.001,  # Store result in min_value
            key="z_sigma"  # Store result in key
        )
    else:
        sigma = None   # Will use sample std as approximation

    alpha = st.selectbox("Significance level (α):", [0.05, 0.01, 0.10], key="z_alpha")  # Store result in alpha

    if st.button("Run Z-test"):  # Check condition
        if len(series) < 2:  # Check condition
            st.error("Need at least 2 data points.")  # Show a red error message
            return  # Exit function early

        if len(series) < 30:  # Check condition
            # Warn if n is small — Z-test assumes large sample
            st.warning(f"⚠️ n={len(series)} < 30. Z-test is designed for large samples. Consider T-test instead.")  # Show a yellow warning message

        # Use known σ if provided, otherwise use sample std dev as approximation
        std_used = sigma if use_known_sigma else series.std(ddof=1)  # Store result in std_used

        n = len(series)   # Sample size

        # Z-statistic formula: z = (x̄ - μ₀) / (σ / √n)
        z_stat = (series.mean() - mu0) / (std_used / np.sqrt(n))  # Compute square root

        # Two-tailed p-value: P(|Z| > |z_stat|) = 2 × P(Z > |z_stat|)
        # sp.norm.cdf gives cumulative probability up to a value
        p_value = 2 * (1 - sp.norm.cdf(abs(z_stat)))  # Compute Normal CDF values

        _display_test_results(  # Execute this statement
            test_name="One-Sample Z-test",  # Store result in test_name
            statistic_name="z-statistic",  # Store result in statistic_name
            statistic=z_stat,  # Store result in statistic
            p_value=p_value,  # Store result in p_value
            df=None,     # Z-test has no degrees of freedom
            alpha=alpha,  # Store result in alpha
            h0=f"H₀: mean of '{col}' = {mu0}",  # Store result in h0
            h1=f"H₁: mean of '{col}' ≠ {mu0}",  # Store result in h1
            extra_info={  # Store result in extra_info
                "Sample Mean (x̄)": round(series.mean(), 4),  # Execute this statement
                "Hypothesized Mean (μ₀)": mu0,  # Execute this statement
                "Difference (x̄ - μ₀)": round(series.mean() - mu0, 4),  # Execute this statement
                "σ used": round(std_used, 4),  # Execute this statement
                "Source of σ": "Known population σ" if use_known_sigma else "Sample std dev (approximation)",  # Execute this statement
                "n": n,  # Execute this statement
                "Z formula applied": f"z = ({series.mean():.2f} - {mu0}) / ({std_used:.2f} / √{n}) = {z_stat:.4f}",  # Store result in "Z formula applied": f"z
            }
        )


def _one_way_anova(df: pd.DataFrame):  # Define the _one_way_anova function
    """
    One-Way ANOVA test.  # Execute this statement
    Tests whether means of 3+ groups are equal.  # Execute this statement

    H₀: μ₁ = μ₂ = ... = μₖ   (all group means equal)  # Store result in H₀: μ₁
    H₁: at least one group mean differs  # Execute this statement

    FIX: Added Tukey HSD post-hoc test — shows WHICH specific pairs of groups differ.  # Execute this statement
    Without post-hoc, ANOVA only says "something is different" without saying what.  # Execute this statement
    """
    st.markdown("### One-Way ANOVA")  # Render formatted markdown text in the Streamlit UI
    st.write("Tests whether 3 or more group means are significantly different.")  # Display text or data in the Streamlit UI

    numeric_cols = get_numeric_columns(df)    # Numeric columns for the outcome variable
    cat_cols = get_categorical_columns(df)     # Categorical columns for grouping

    if not cat_cols:  # Check condition
        st.error("One-Way ANOVA requires a categorical grouping column.")  # Show a red error message
        return  # Exit function early
    if not numeric_cols:  # Check condition
        st.error("No numeric columns available.")  # Show a red error message
        return  # Exit function early

    group_col = st.selectbox("Grouping column:", cat_cols, key="anova1_grp")     # The groups
    num_col = st.selectbox("Numeric column:", numeric_cols, key="anova1_num")     # The outcome
    alpha = st.selectbox("Significance level (α):", [0.05, 0.01, 0.10], key="anova1_alpha")  # Store result in alpha

    if st.button("Run One-Way ANOVA"):  # Check condition
        group_labels = df[group_col].dropna().unique()   # All unique group names
        groups = []        # List to store data for each group
        valid_labels = []  # Group names that have enough data

        # Build list of groups — only include groups with at least 2 values
        for g in group_labels:  # Loop over each item
            grp_data = drop_missing(df[df[group_col] == g][num_col])  # Data for this group
            if len(grp_data) >= 2:             # Need at least 2 values per group
                groups.append(grp_data)        # Add group data to list
                valid_labels.append(g)         # Record the group name

        if len(groups) < 2:  # Check condition
            st.error("Need at least 2 groups with 2+ values each.")  # Show a red error message
            return  # Exit function early

        # sp.f_oneway performs one-way ANOVA
        # * unpacks the list so each group is a separate argument
        f_stat, p_value = sp.f_oneway(*groups)  # One-way ANOVA F-test across multiple groups

        k = len(groups)                          # Number of groups
        n_total = sum(len(g) for g in groups)    # Total number of observations across all groups
        df_between = k - 1                       # Between-group degrees of freedom = k-1
        df_within = n_total - k                  # Within-group degrees of freedom = N-k

        _display_test_results(  # Execute this statement
            test_name="One-Way ANOVA",  # Store result in test_name
            statistic_name="F-statistic",  # Store result in statistic_name
            statistic=f_stat,  # Store result in statistic
            p_value=p_value,  # Store result in p_value
            df=f"{df_between}, {df_within}",   # Show both dfs as "df1, df2"
            alpha=alpha,  # Store result in alpha
            h0=f"H₀: All group means of '{num_col}' are equal",  # Store result in h0
            h1=f"H₁: At least one group mean is different",  # Store result in h1
            extra_info={  # Store result in extra_info
                "Number of groups (k)": k,  # Execute this statement
                "Total observations (N)": n_total,  # Execute this statement
                "df_between = k-1": df_between,  # Store result in "df_between
                "df_within = N-k": df_within,  # Store result in "df_within
            }
        )

        # Show a table of group means for comparison
        st.markdown("**Group Means:**")  # Render formatted markdown text in the Streamlit UI
        means_df = pd.DataFrame({  # Create DataFrame from dictionary or array
            "Group": [str(g) for g in valid_labels],              # Group names
            "Mean": [round(grp.mean(), 4) for grp in groups],     # Mean per group
            "Std Dev": [round(grp.std(ddof=1), 4) for grp in groups],  # Std dev per group
            "n": [len(grp) for grp in groups],                    # Count per group
        })  # Execute this statement
        st.dataframe(means_df, use_container_width=True, hide_index=True)  # Render an interactive data table

        # FIX: Tukey HSD post-hoc test
        # Only runs if ANOVA was significant — no point running post-hoc otherwise
        if p_value < alpha:  # Check condition
            st.markdown("---")  # Render formatted markdown text in the Streamlit UI
            st.markdown("#### 🔍 Tukey HSD Post-hoc Test")  # Render formatted markdown text in the Streamlit UI
            st.write("""  # Display text or data in the Streamlit UI
            ANOVA tells you *that* at least one group differs — but not *which* ones.  # Execute this statement
            Tukey HSD (Honestly Significant Difference) tests every pair of groups.  # Execute this statement
            It controls for multiple comparisons to avoid false positives.  # Execute this statement
            """)
            try:
                # Combine all group values into one array
                all_values = np.concatenate(groups)  # Join multiple arrays into one

                # Build a matching array of group labels (one label per data point)
                all_labels = np.concatenate(  # Join multiple arrays into one
                    [[str(lbl)] * len(grp) for lbl, grp in zip(valid_labels, groups)]  # Execute this statement
                )

                # Run Tukey HSD: compares every pair of groups
                tukey = pairwise_tukeyhsd(all_values, all_labels, alpha=alpha)  # Store result in tukey

                # Convert Tukey result table to a pandas DataFrame
                tukey_df = pd.DataFrame(  # Create DataFrame from dictionary or array
                    data=tukey._results_table.data[1:],        # Rows (skip header row)
                    columns=tukey._results_table.data[0]       # Column names from header
                )
                # Rename columns to be clearer
                tukey_df.columns = ["Group 1", "Group 2", "Mean Diff", "p-adj", "Lower CI", "Upper CI", "Reject H0"]  # Store result in columns

                # Round numeric columns for clean display
                for c in ["Mean Diff", "p-adj", "Lower CI", "Upper CI"]:  # Loop over each item
                    tukey_df[c] = tukey_df[c].apply(lambda x: round(float(x), 4))  # Store result in tukey_df

                st.dataframe(tukey_df, use_container_width=True, hide_index=True)  # Render an interactive data table

                # Plain English summary of which pairs are significantly different
                st.markdown("**Summary of significant differences:**")  # Render formatted markdown text in the Streamlit UI
                sig_pairs = tukey_df[tukey_df["Reject H0"] == True]   # Only significant pairs

                if len(sig_pairs) > 0:  # Check condition
                    for _, row in sig_pairs.iterrows():  # Loop over each item
                        # Show each significant pair with mean difference and adjusted p-value
                        st.success(  # Show a green success message
                            f"✅ **{row['Group 1']}** vs **{row['Group 2']}**: "  # Execute this statement
                            f"mean difference = {row['Mean Diff']:.2f}, "  # Store result in f"mean difference
                            f"adjusted p = {row['p-adj']:.4f} → Significantly different"  # Store result in f"adjusted p
                        )
                else:
                    # ANOVA was significant but Tukey found no specific pair — edge case
                    st.info("No specific pair found to be significantly different after Tukey correction.")  # Show an informational blue message box

            except Exception as e:  # Handle any error from the try block
                st.warning(f"Tukey HSD could not run: {e}")   # Show error without crashing

        else:
            # ANOVA not significant — no need for post-hoc
            st.info("ℹ️ ANOVA result was not significant — post-hoc test is not needed.")  # Show an informational blue message box


def _two_way_anova(df: pd.DataFrame):  # Define the _two_way_anova function
    """
    Two-Way ANOVA using statsmodels OLS.  # Execute this statement
    Tests effect of two categorical factors and their interaction on a numeric outcome.  # Execute this statement

    Tests three things simultaneously:  # Execute this statement
    1. Main effect of Factor 1  # Execute this statement
    2. Main effect of Factor 2  # Execute this statement
    3. Interaction effect of Factor1 × Factor2  # Execute this statement
    """
    st.markdown("### Two-Way ANOVA")  # Render formatted markdown text in the Streamlit UI
    st.write("Tests the effect of two categorical variables and their interaction on a numeric outcome.")  # Display text or data in the Streamlit UI

    numeric_cols = get_numeric_columns(df)   # Numeric outcome variable options
    cat_cols = get_categorical_columns(df)   # Categorical factor options

    if len(cat_cols) < 2:  # Check condition
        st.error("Two-Way ANOVA requires at least 2 categorical columns.")  # Show a red error message
        return  # Exit function early
    if not numeric_cols:  # Check condition
        st.error("No numeric columns available.")  # Show a red error message
        return  # Exit function early

    factor1 = st.selectbox("Factor 1 (categorical):", cat_cols, key="anova2_f1")   # First factor
    # Exclude factor1 from the options for factor2 to prevent duplicate selection
    factor2 = st.selectbox("Factor 2 (categorical):",  # Store result in factor2
                            [c for c in cat_cols if c != factor1], key="anova2_f2")  # Store result in 
    outcome = st.selectbox("Outcome (numeric):", numeric_cols, key="anova2_out")    # Dependent variable

    if st.button("Run Two-Way ANOVA"):  # Check condition
        try:
            # Build a clean DataFrame with only the needed columns, removing rows with NaN
            model_df = df[[factor1, factor2, outcome]].dropna().copy()  # Copy so we can rename columns safely

            if len(model_df) < 10:  # Check condition
                st.error("Need at least 10 complete rows for Two-Way ANOVA.")  # Show a red error message
                return  # Exit function early

            # Rename columns to simple safe names for the OLS formula
            # This avoids "invalid syntax" errors caused by spaces, backticks,
            # or special characters in column names inside the statsmodels formula parser
            safe_f1 = "Factor1"       # Safe name for factor 1 — no spaces or special chars
            safe_f2 = "Factor2"       # Safe name for factor 2
            safe_out = "Outcome"      # Safe name for outcome variable
            model_df.columns = [safe_f1, safe_f2, safe_out]  # Apply the safe column names

            # Build the OLS formula using safe column names
            # C() tells statsmodels to treat the variable as categorical (not numeric)
            # Factor1:Factor2 adds the interaction term between the two factors
            formula = f"{safe_out} ~ C({safe_f1}) + C({safe_f2}) + C({safe_f1}):C({safe_f2})"  # Clean formula without backticks

            # Fit the Ordinary Least Squares (OLS) linear model
            model = ols(formula, data=model_df).fit()  # Fit model to renamed data

            # Compute the ANOVA table using Type II sum of squares
            # Type II SS tests each factor controlling for the other factor
            anova_table = sm.stats.anova_lm(model, typ=2)  # Store result in anova_table

            # Rename the index back to original column names for display
            # So the user sees "Department" not "Factor1" in the results table
            index_map = {                                        # Mapping from safe names back to original
                f"C({safe_f1})": f"C({factor1})",               # Factor 1 row label
                f"C({safe_f2})": f"C({factor2})",               # Factor 2 row label
                f"C({safe_f1}):C({safe_f2})": f"C({factor1}):C({factor2})",  # Interaction row label
            }
            anova_table.index = [index_map.get(i, i) for i in anova_table.index]  # Apply rename

            st.markdown("**ANOVA Table (Type II SS):**")  # Render formatted markdown text in the Streamlit UI
            st.dataframe(anova_table.round(4), use_container_width=True)  # Display ANOVA table

            # Interpret each row's p-value in plain English
            st.markdown("**Interpretation:**")  # Render formatted markdown text in the Streamlit UI
            for idx in anova_table.index[:-1]:   # Skip last row (Residual)
                p = anova_table.loc[idx, "PR(>F)"]   # p-value for this factor
                F = anova_table.loc[idx, "F"]         # F-statistic for this factor
                if p < 0.05:  # Check condition
                    st.success(f"✅ `{idx}`: F={F:.4f}, p={format_pvalue(p)} → **Significant effect**")  # Show a green success message
                else:
                    st.info(f"ℹ️ `{idx}`: F={F:.4f}, p={format_pvalue(p)} → No significant effect")  # Show an informational blue message box

        except Exception as e:  # Handle any error from the try block
            st.error(f"Two-Way ANOVA failed: {e}")  # Show a red error message
            st.warning("Make sure each factor has ≥2 levels and there are enough observations per cell.")  # Show a yellow warning message


def _chi_square_test(df: pd.DataFrame):  # Define the _chi_square_test function
    """
    Chi-Square Test of Independence.  # Execute this statement
    Tests whether two categorical variables are related or independent.  # Execute this statement

    H₀: the two variables are independent (no association)  # Execute this statement
    H₁: the two variables are NOT independent (there is an association)  # Execute this statement
    """
    st.markdown("### Chi-Square Test of Independence")  # Render formatted markdown text in the Streamlit UI
    st.write("Tests whether two categorical variables are associated.")  # Display text or data in the Streamlit UI

    cat_cols = get_categorical_columns(df)   # Only categorical columns make sense here

    if len(cat_cols) < 2:  # Check condition
        st.error("Chi-Square test requires at least 2 categorical columns.")  # Show a red error message
        return  # Exit function early

    var1 = st.selectbox("Variable 1:", cat_cols, key="chi_v1")   # First categorical variable
    var2 = st.selectbox("Variable 2:", [c for c in cat_cols if c != var1], key="chi_v2")  # Second
    alpha = st.selectbox("Significance level (α):", [0.05, 0.01, 0.10], key="chi_alpha")  # Store result in alpha

    if st.button("Run Chi-Square Test"):  # Check condition
        # Drop rows where either variable is missing
        clean_df = df[[var1, var2]].dropna()  # Remove rows with NaN values

        if len(clean_df) < 5:  # Check condition
            st.error("Need at least 5 complete observations.")  # Show a red error message
            return  # Exit function early

        # pd.crosstab creates a frequency table of the two variables
        contingency_table = pd.crosstab(clean_df[var1], clean_df[var2])  # Create contingency table of two categorical variables

        st.markdown("**Contingency Table (Observed Frequencies):**")  # Render formatted markdown text in the Streamlit UI
        st.dataframe(contingency_table, use_container_width=True)  # Show observed counts

        # sp.chi2_contingency performs the chi-square test
        # Returns: chi2 statistic, p-value, degrees of freedom, expected frequencies
        chi2, p_value, dof, expected = sp.chi2_contingency(contingency_table)  # Chi-square test of independence

        # Chi-square assumption: expected frequencies should be ≥ 5 in each cell
        low_expected = (expected < 5).sum()   # Count cells with expected < 5
        if low_expected > 0:  # Check condition
            st.warning(  # Show a yellow warning message
                f"⚠️ {low_expected} cell(s) have expected frequency < 5. "  # Execute this statement
                "Chi-square assumption violated — results may not be reliable. "  # Execute this statement
                "Consider merging categories or using Fisher's exact test."  # Execute this statement
            )

        _display_test_results(  # Execute this statement
            test_name="Chi-Square Test of Independence",  # Store result in test_name
            statistic_name="χ² statistic",  # Store result in statistic_name
            statistic=chi2,  # Store result in statistic
            p_value=p_value,  # Store result in p_value
            df=dof,    # Degrees of freedom = (rows-1) × (cols-1)
            alpha=alpha,  # Store result in alpha
            h0=f"H₀: '{var1}' and '{var2}' are independent",  # Store result in h0
            h1=f"H₁: '{var1}' and '{var2}' are NOT independent (associated)",  # Store result in h1
            extra_info={  # Store result in extra_info
                "dof = (rows-1) × (cols-1)": f"({contingency_table.shape[0]-1}) × ({contingency_table.shape[1]-1}) = {dof}",  # Store result in "dof
                "n (total observations)": len(clean_df),  # Execute this statement
                "Cells with expected < 5": int(low_expected),  # Execute this statement
            }
        )

        # Show expected frequencies for inspection
        with st.expander("📋 Expected Frequencies (under H₀)"):  # Open context manager
            expected_df = pd.DataFrame(  # Create DataFrame from dictionary or array
                expected,                              # Expected counts as array
                index=contingency_table.index,         # Same row labels as observed table
                columns=contingency_table.columns      # Same column labels as observed table
            ).round(2)  # Execute this statement
            st.dataframe(expected_df, use_container_width=True)  # Render an interactive data table


# =====================================================================
# Shared display and utility functions
# =====================================================================

def _display_test_results(test_name, statistic_name, statistic, p_value,  # Define the _display_test_results function
                           df, alpha, h0, h1, extra_info: dict):  # Execute this statement
    """
    Display hypothesis test results in a standardized, readable format.  # Execute this statement
    Shows: hypotheses, test statistic, p-value, df, decision, and extra info.  # Execute this statement

    Parameters:  # Execute this statement
        test_name (str): Name of the test for the expander title  # Execute this statement
        statistic_name (str): Label for the test statistic (e.g. 't-statistic')  # Execute this statement
        statistic (float): The computed test statistic value  # Execute this statement
        p_value (float): The computed p-value  # Execute this statement
        df: Degrees of freedom — int, string like "3, 196", or None  # Execute this statement
        alpha (float): Significance level chosen by user  # Execute this statement
        h0 (str): Null hypothesis statement  # Execute this statement
        h1 (str): Alternative hypothesis statement  # Execute this statement
        extra_info (dict): Additional key-value pairs to display (e.g. sample means)  # Execute this statement
    """
    with st.expander(f"📋 Results: {test_name}", expanded=True):  # Open context manager
        st.markdown(f"**{h0}**")   # Show null hypothesis
        st.markdown(f"**{h1}**")   # Show alternative hypothesis
        st.markdown("---")  # Render formatted markdown text in the Streamlit UI

        # Display three key numbers as metric widgets side by side
        col1, col2, col3 = st.columns(3)  # Store result in col1, col2, col3
        with col1:  # Open context manager
            st.metric(statistic_name, f"{statistic:.4f}")   # Test statistic value
        with col2:  # Open context manager
            st.metric("P-value", format_pvalue(p_value))    # Formatted p-value
        with col3:  # Open context manager
            if df is not None:  # Check condition
                st.metric("Degrees of Freedom", str(df))    # df (if applicable)

        st.markdown("---")  # Render formatted markdown text in the Streamlit UI
        st.markdown(f"**Significance Level (α):** {alpha}")  # Show chosen alpha

        # Decision: reject or fail to reject H₀ based on p-value vs alpha
        if p_value < alpha:  # Check condition
            # p < α → statistically significant result → reject H₀
            st.error(  # Show a red error message
                f"🔴 **Reject H₀** — p-value ({format_pvalue(p_value)}) < α ({alpha})\n\n"  # Execute this statement
                f"There IS a statistically significant result.\n{h1}"  # Execute this statement
            )
        else:
            # p ≥ α → not enough evidence → fail to reject H₀
            st.success(  # Show a green success message
                f"🟢 **Fail to Reject H₀** — p-value ({format_pvalue(p_value)}) ≥ α ({alpha})\n\n"  # Execute this statement
                f"There is NOT enough evidence to reject:\n{h0}"  # Execute this statement
            )

        # Show additional details (sample means, n, etc.)
        if extra_info:  # Check condition
            st.markdown("**Additional Information:**")  # Render formatted markdown text in the Streamlit UI
            for key, value in extra_info.items():  # Loop over each item
                st.write(f"- {key}: `{value}`")   # Each info item on its own line


def _check_normality(series: pd.Series, col_name: str):  # Define the _check_normality function
    """
    Check whether a sample is approximately normally distributed.  # Execute this statement
    Uses Shapiro-Wilk test (reliable for n ≤ 5000).  # Execute this statement
    T-test assumes normality, especially for small samples.  # Execute this statement

    Parameters:  # Execute this statement
        series (pd.Series): Cleaned numeric data to test  # Pandas data operation
        col_name (str): Column or group name for display  # Execute this statement
    """
    with st.expander(f"🔍 Normality Check for '{col_name}'"):  # Open context manager
        n = len(series)   # Sample size

        if n > 5000:  # Check condition
            # For large samples, CLT guarantees approximately normal sampling distribution
            # Shapiro-Wilk also becomes unreliable for very large n
            st.info(f"n={n} is large — by CLT the t-test is robust even for non-normal data.")  # Show an informational blue message box
            return  # Exit function early

        # Shapiro-Wilk test: tests if data comes from a normal distribution
        # W close to 1 and p > 0.05 = data is approximately normal
        stat, p = sp.shapiro(series)  # Shapiro-Wilk normality test

        st.write(f"**Shapiro-Wilk Test:** W = {stat:.4f}, p = {format_pvalue(p)}")  # Display text or data in the Streamlit UI

        if p > 0.05:  # Check condition
            # Cannot reject normality — t-test assumption is satisfied
            st.success("✅ Data appears normally distributed (p > 0.05). T-test assumption met.")  # Show a green success message
        else:
            # Normality is rejected — warn the user but explain when it's still OK
            st.warning(  # Show a yellow warning message
                f"⚠️ Data may NOT be normally distributed (p ≤ 0.05). "  # Execute this statement
                f"T-test is still valid for n={n} ≥ 30 by Central Limit Theorem. "  # Store result in f"T-test is still valid for n
                f"For small samples (n < 30), consider Mann-Whitney U test instead."  # Execute this statement
            )


# =====================================================================
# Pearson Correlation Coefficient
# Added per project requirements: uses real data, shows formula,
# explains when it cannot be computed, includes significance test
# =====================================================================

def _pearson_correlation(df: pd.DataFrame):
    """
    Pearson Correlation Coefficient.

    Measures the strength and direction of the LINEAR relationship
    between two numeric variables. Range: -1 to +1.

    Formula:
        r = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / √[Σ(xᵢ - x̄)² × Σ(yᵢ - ȳ)²]

    Interpretation:
        r = +1 : perfect positive linear relationship
        r =  0 : no linear relationship
        r = -1 : perfect negative linear relationship

    Hypothesis test for significance:
        H₀: ρ = 0  (no linear correlation in the population)
        H₁: ρ ≠ 0  (there IS a linear correlation)

    Also includes:
        - Full pairwise correlation matrix for all numeric columns
        - Scatter plot of the selected pair with regression line
        - Assumptions check (normality, linearity, outliers)
        - Clear explanation when correlation cannot be computed
    """
    st.markdown("### Pearson Correlation Coefficient")  # Display sub-heading for this test

    # Explain what Pearson correlation measures before showing controls
    st.info("""
    **What is Pearson Correlation?**
    It measures how strongly two numeric variables move together in a LINEAR pattern.
    - **r > 0**: when X increases, Y tends to increase (positive relationship)
    - **r < 0**: when X increases, Y tends to decrease (negative relationship)
    - **r = 0**: no linear relationship (but non-linear relationship may still exist)

    ⚠️ Pearson only captures **linear** relationships. Always check the scatter plot.
    """)

    numeric_cols = get_numeric_columns(df)  # Get all numeric columns from the dataset

    # Need at least 2 numeric columns to compute a correlation between them
    if len(numeric_cols) < 2:
        st.error("""
        ❌ **Pearson Correlation cannot be computed.**

        **Reason:** Correlation requires at least 2 numeric columns.
        Your dataset has fewer than 2 numeric columns.

        **What to do:** Upload a dataset with at least 2 numeric variables
        (e.g. Age, Salary, Score, Height, Weight).
        """)
        return  # Exit — nothing to compute with fewer than 2 numeric columns

    # ---- SECTION 1: Full pairwise correlation matrix ----
    st.markdown("#### 📋 Pairwise Correlation Matrix (all numeric columns)")  # Section heading

    # Compute the full Pearson correlation matrix for all numeric columns at once
    # df[numeric_cols].corr() uses Pearson method by default
    # It drops NaN pairs automatically (pairwise complete observations)
    corr_matrix = df[numeric_cols].corr(method="pearson")  # Compute pairwise Pearson r values

    # Display the correlation matrix as an interactive table
    st.dataframe(
        corr_matrix.round(4),               # Round to 4 decimal places for readability
        use_container_width=True            # Stretch table to full width of the page
    )

    # Explain the matrix to the user
    st.caption(
        "Each cell shows r between that pair. Diagonal is always 1.0 (a variable is perfectly correlated with itself)."
    )

    # ---- SECTION 2: Select a specific pair to analyse in detail ----
    st.markdown("---")  # Visual divider between sections
    st.markdown("#### 🔍 Detailed Analysis — Select a Pair")  # Section heading

    # Let user select the X variable (first column in the pair)
    x_col = st.selectbox(
        "Select X variable:",       # Dropdown label
        numeric_cols,               # All numeric columns are valid options
        key="pearson_x"             # Unique key prevents widget conflict
    )

    # Let user select the Y variable (second column) — exclude X to prevent self-correlation
    y_col = st.selectbox(
        "Select Y variable:",                                    # Dropdown label
        [c for c in numeric_cols if c != x_col],                # Exclude the X column from options
        key="pearson_y"                                          # Unique key
    )

    # Significance level for the hypothesis test
    alpha = st.selectbox("Significance level (α):", [0.05, 0.01, 0.10], key="pearson_alpha")

    if st.button("Compute Pearson Correlation"):  # Only run when user clicks the button

        # Drop rows where EITHER column has a missing value — both values needed for a pair
        pair_df = df[[x_col, y_col]].dropna()  # Keep only rows with both values present

        n = len(pair_df)  # Number of complete observation pairs available

        # ---- Check: need at least 3 observations ----
        if n < 3:
            st.error(f"""
            ❌ **Pearson Correlation cannot be computed for '{x_col}' vs '{y_col}'.**

            **Reason:** After removing missing values, only {n} complete pair(s) remain.
            Pearson correlation requires at least 3 observations to be meaningful.

            **What to do:** Choose columns with more non-missing overlapping data.
            """)
            return  # Exit — not enough data to compute a meaningful correlation

        # ---- Check: constant column (zero variance) ----
        # If one column has the same value for all rows, correlation is undefined (division by zero)
        if pair_df[x_col].std() == 0:
            st.error(f"""
            ❌ **Pearson Correlation cannot be computed.**

            **Reason:** Column '{x_col}' has zero variance — all values are identical ({pair_df[x_col].iloc[0]}).
            The Pearson formula requires dividing by the standard deviation.
            Division by zero is undefined, so r cannot be calculated.

            **Formula reference:**
            r = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / (σₓ × σᵧ × n)
            When σₓ = 0, the denominator is 0 → undefined.

            **What to do:** Choose a column that has variation in its values.
            """)
            return  # Exit — zero variance makes the formula undefined

        if pair_df[y_col].std() == 0:
            st.error(f"""
            ❌ **Pearson Correlation cannot be computed.**

            **Reason:** Column '{y_col}' has zero variance — all values are identical ({pair_df[y_col].iloc[0]}).
            When σᵧ = 0, the denominator of the Pearson formula is 0 → undefined.

            **What to do:** Choose a column that has variation in its values.
            """)
            return  # Exit — zero variance in Y column

        # ---- Compute Pearson r and p-value ----
        # sp.pearsonr returns (r, p_value) where:
        #   r = Pearson correlation coefficient (-1 to +1)
        #   p_value = two-tailed significance test (H0: ρ = 0)
        r, p_value = sp.pearsonr(pair_df[x_col], pair_df[y_col])  # Compute r and significance

        # Degrees of freedom for Pearson correlation = n - 2
        # (we estimate 2 parameters: means of x and y)
        df_val = n - 2  # Degrees of freedom

        # t-statistic for testing H0: ρ = 0
        # Formula: t = r × √(n-2) / √(1-r²)
        # This converts r to a t-distribution for significance testing
        if abs(r) < 1.0:  # Guard: avoid division by zero when r = exactly ±1
            t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)  # Compute t-statistic from r
        else:
            t_stat = float('inf')  # Perfect correlation → infinite t-statistic

        # R-squared = proportion of variance in Y explained by X
        r_squared = r ** 2  # Square the correlation coefficient

        # ---- Display the formula ----
        with st.expander("📖 Formula Used", expanded=True):  # Collapsible formula section
            st.latex(r"r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i-\bar{x})^2 \cdot \sum(y_i-\bar{y})^2}}")  # Pearson r formula
            st.latex(r"r = \frac{n\sum x_i y_i - \sum x_i \sum y_i}{\sqrt{[n\sum x_i^2 - (\sum x_i)^2][n\sum y_i^2 - (\sum y_i)^2]}}")  # Equivalent computational formula
            st.latex(r"t = \frac{r\sqrt{n-2}}{\sqrt{1-r^2}}, \quad df = n - 2")  # t-test formula for significance
            st.latex(r"R^2 = r^2 \quad \text{(proportion of variance explained)}")  # R-squared formula
            st.write("""
            **Where:**
            - xᵢ, yᵢ = individual data point values
            - x̄, ȳ = sample means of X and Y
            - n = number of complete observation pairs
            - r = Pearson correlation coefficient
            - t = test statistic (follows t-distribution under H₀)
            - df = degrees of freedom = n - 2
            """)

        # ---- Display main results as metrics ----
        st.markdown("---")  # Visual divider
        st.markdown("**Results from your data:**")  # Results heading

        c1, c2, c3, c4 = st.columns(4)  # Four metric columns side by side
        with c1:
            st.metric("r (correlation)", f"{r:.4f}")  # The Pearson r value
        with c2:
            st.metric("r² (explained var.)", f"{r_squared:.4f}")  # Proportion of variance explained
        with c3:
            st.metric("t-statistic", f"{t_stat:.4f}")  # t-statistic for significance test
        with c4:
            st.metric("P-value", format_pvalue(p_value))  # Formatted p-value

        # ---- Hypothesis test decision ----
        st.markdown(f"**H₀:** ρ = 0 (no linear correlation between '{x_col}' and '{y_col}')")  # Null hypothesis
        st.markdown(f"**H₁:** ρ ≠ 0 (there IS a linear correlation)")  # Alternative hypothesis
        st.markdown(f"**df = n - 2 = {n} - 2 = {df_val}**")  # Show df calculation

        if p_value < alpha:  # Check if result is statistically significant
            st.error(
                f"🔴 **Reject H₀** — p = {format_pvalue(p_value)} < α ({alpha})\n\n"
                f"The correlation r = {r:.4f} is **statistically significant**.\n"
                f"There IS a real linear relationship between '{x_col}' and '{y_col}'."
            )
        else:
            st.success(
                f"🟢 **Fail to Reject H₀** — p = {format_pvalue(p_value)} ≥ α ({alpha})\n\n"
                f"The correlation r = {r:.4f} is **NOT statistically significant**.\n"
                f"There is insufficient evidence of a linear relationship."
            )

        # ---- Interpret the strength of r in plain English ----
        st.markdown("#### 📊 Interpretation of r = " + f"{r:.4f}")  # Strength heading
        abs_r = abs(r)  # Use absolute value to assess strength regardless of direction

        if abs_r >= 0.9:
            strength = "Very strong"  # Very high correlation
            color = "🟣"             # Color indicator
        elif abs_r >= 0.7:
            strength = "Strong"       # High correlation
            color = "🔵"
        elif abs_r >= 0.5:
            strength = "Moderate"     # Moderate correlation
            color = "🟡"
        elif abs_r >= 0.3:
            strength = "Weak"         # Low correlation
            color = "🟠"
        else:
            strength = "Very weak or negligible"  # Negligible correlation
            color = "⚪"

        direction = "positive" if r > 0 else "negative"  # Direction of the relationship

        st.info(f"""
        {color} **{strength} {direction} correlation** (r = {r:.4f})

        **r² = {r_squared:.4f}** → '{x_col}' explains **{r_squared*100:.1f}%** of the variance in '{y_col}'.

        **Plain English:** {"As " + x_col + " increases, " + y_col + " tends to " + ("increase" if r > 0 else "decrease") + "." if abs_r >= 0.3 else f"There is no meaningful linear trend between {x_col} and {y_col}."}

        **Correlation strength guide:**
        - |r| ≥ 0.9 → Very strong | |r| ≥ 0.7 → Strong | |r| ≥ 0.5 → Moderate
        - |r| ≥ 0.3 → Weak | |r| < 0.3 → Negligible
        """)

        # ---- Additional statistics ----
        with st.expander("📋 Detailed Statistics", expanded=True):  # Expandable details section
            stats_df = pd.DataFrame({  # Build summary table of all computed values
                "Statistic": [
                    "n (complete pairs)",       # How many pairs were used
                    f"Mean of {x_col}",         # Mean of X
                    f"Mean of {y_col}",         # Mean of Y
                    f"Std Dev of {x_col}",      # Standard deviation of X
                    f"Std Dev of {y_col}",      # Standard deviation of Y
                    "Pearson r",                # The correlation coefficient
                    "r² (coefficient of determination)",  # Variance explained
                    "t-statistic",              # Test statistic
                    "Degrees of freedom (df)",  # n - 2
                    "p-value (two-tailed)",     # Significance
                    "Significance level (α)",   # User-chosen alpha
                ],
                "Value": [
                    n,                                          # Number of pairs
                    round(pair_df[x_col].mean(), 4),           # Mean of X
                    round(pair_df[y_col].mean(), 4),           # Mean of Y
                    round(pair_df[x_col].std(ddof=1), 4),      # Std dev of X (sample)
                    round(pair_df[y_col].std(ddof=1), 4),      # Std dev of Y (sample)
                    round(r, 4),                               # Pearson r
                    round(r_squared, 4),                       # R-squared
                    round(t_stat, 4),                          # t-statistic
                    df_val,                                    # Degrees of freedom
                    format_pvalue(p_value),                    # Formatted p-value
                    alpha,                                     # Significance level
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)  # Display table

        # ---- Scatter plot with regression line ----
        st.markdown("#### 📈 Scatter Plot with Regression Line")  # Plot heading

        import matplotlib.pyplot as plt  # Import matplotlib for the scatter plot
        fig, ax = plt.subplots(figsize=(9, 5))  # Create figure and axis

        # Draw scatter plot of the actual data points
        ax.scatter(
            pair_df[x_col],         # X-axis values from selected column
            pair_df[y_col],         # Y-axis values from selected column
            alpha=0.6,              # Semi-transparent points to show overlapping density
            color="#4C72B0",        # Blue color for data points
            edgecolors="white",     # White border on each point for clarity
            s=50,                   # Point size
            label="Data points"     # Legend label
        )

        # Compute and draw the OLS regression line (y = mx + b)
        # np.polyfit degree 1 fits a straight line: returns [slope, intercept]
        m, b = np.polyfit(pair_df[x_col], pair_df[y_col], 1)  # Fit linear regression line

        # Create x values spanning the full data range for drawing the line
        x_line = np.linspace(pair_df[x_col].min(), pair_df[x_col].max(), 200)  # Smooth x range
        y_line = m * x_line + b  # Compute corresponding y values using the fitted line

        ax.plot(
            x_line, y_line,                                         # Plot the regression line
            color="#C44E52",                                        # Red color for visibility
            linewidth=2,                                            # Line thickness
            label=f"Regression line (r={r:.3f})"                   # Legend shows r value
        )

        ax.set_title(f"Scatter Plot: {x_col} vs {y_col}\nr = {r:.4f}, r² = {r_squared:.4f}, p = {format_pvalue(p_value)}")  # Title shows key stats
        ax.set_xlabel(x_col)    # Label x-axis with column name
        ax.set_ylabel(y_col)    # Label y-axis with column name
        ax.legend(fontsize=9)   # Show the legend
        ax.grid(alpha=0.3)      # Add faint grid lines

        plt.tight_layout()      # Prevent label overlap
        st.pyplot(fig)          # Render the figure in Streamlit
        plt.close(fig)          # Free memory after rendering

        # ---- Assumptions check ----
        st.markdown("#### ✅ Assumptions Check")  # Assumptions heading
        st.write("""
        Pearson correlation has 4 key assumptions. Violations affect the validity of results:
        """)

        # Assumption 1: Both variables must be numeric (continuous)
        st.markdown("**1. Both variables must be numeric (continuous)** — ✅ Confirmed (you selected numeric columns)")

        # Assumption 2: Linear relationship — check via scatter plot
        st.markdown("**2. Linear relationship** — Check the scatter plot above. If the points follow a curve rather than a straight line, use Spearman rank correlation instead.")

        # Assumption 3: Normality — check with Shapiro-Wilk
        st.markdown("**3. Approximate normality of both variables:**")
        for col_name, col_data in [(x_col, pair_df[x_col]), (y_col, pair_df[y_col])]:
            if len(col_data) <= 5000:  # Shapiro-Wilk is reliable up to n=5000
                w_stat, sw_p = sp.shapiro(col_data)  # Run Shapiro-Wilk normality test
                normal_result = "✅ Approximately normal" if sw_p > 0.05 else "⚠️ NOT normal (consider Spearman)"  # Interpret result
                st.write(f"   - {col_name}: Shapiro-Wilk W={w_stat:.4f}, p={format_pvalue(sw_p)} → {normal_result}")
            else:
                st.write(f"   - {col_name}: n={len(col_data)} > 5000 → Normality assumed by CLT")  # Large samples: CLT applies

        # Assumption 4: No extreme outliers — check IQR
        st.markdown("**4. No extreme outliers:**")
        for col_name, col_data in [(x_col, pair_df[x_col]), (y_col, pair_df[y_col])]:
            q1 = col_data.quantile(0.25)   # First quartile
            q3 = col_data.quantile(0.75)   # Third quartile
            iqr = q3 - q1                  # Interquartile range
            lower_fence = q1 - 3 * iqr    # Extreme outlier lower threshold (3×IQR)
            upper_fence = q3 + 3 * iqr    # Extreme outlier upper threshold (3×IQR)
            n_outliers = ((col_data < lower_fence) | (col_data > upper_fence)).sum()  # Count extreme outliers
            outlier_result = "✅ No extreme outliers" if n_outliers == 0 else f"⚠️ {n_outliers} extreme outlier(s) detected"  # Interpret
            st.write(f"   - {col_name}: {outlier_result} (using 3×IQR rule)")

        # Warn about what Pearson cannot detect
        st.warning("""
        ⚠️ **Important limitation:** Pearson r only measures LINEAR relationships.
        A result of r ≈ 0 does NOT mean the variables are unrelated —
        they may have a strong non-linear (e.g. curved, U-shaped) relationship.
        Always inspect the scatter plot to confirm the relationship is actually linear.
        """)
