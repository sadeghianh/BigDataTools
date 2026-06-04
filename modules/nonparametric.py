# =========================
# modules/nonparametric.py
# Non-parametric tests + Normality test + Test Advisor
# All tests use REAL data from the uploaded dataset.
# Tests included:
#   - Normality Test (Shapiro-Wilk + D'Agostino)
#   - Mann-Whitney U (non-parametric two-sample)
#   - Wilcoxon Signed-Rank (non-parametric paired)
#   - Kruskal-Wallis (non-parametric ANOVA)
#   - Fisher's Exact Test (2x2 categorical, alternative to chi-square)
# =========================

import pandas as pd               # Import pandas for DataFrame and Series operations
import numpy as np                # Import numpy for numerical operations
import scipy.stats as sp          # Import scipy.stats for all statistical tests
import streamlit as st            # Import streamlit for UI rendering
from utils.helpers import (       # Import shared utility functions
    get_numeric_columns,          # Returns list of numeric column names
    get_categorical_columns,      # Returns list of categorical column names
    drop_missing,                 # Removes NaN values from a Series
    format_pvalue,                # Formats p-value for clean display
    label_with_unit,              # Builds axis label with unit
)


# =====================================================================
# NORMALITY TEST
# =====================================================================

def render_normality_test(df: pd.DataFrame):
    """
    Test whether a numeric column follows a normal distribution.
    Uses two complementary tests:
      - Shapiro-Wilk (best for small/medium samples, n < 5000)
      - D'Agostino-Pearson (based on skewness and kurtosis)

    Why this matters: many tests (t-test, ANOVA, Pearson) ASSUME normality.
    This test tells you whether those assumptions are met, or whether you
    should use a non-parametric alternative instead.

    H0: The data IS normally distributed
    H1: The data is NOT normally distributed
    """
    st.markdown("### Normality Test (Shapiro-Wilk & D'Agostino)")  # Sub-heading
    st.write("Checks whether a numeric variable follows a normal distribution. "  # Display explanatory text in the UI
             "This determines whether you can use parametric tests (t-test, ANOVA) "
             "or should use non-parametric alternatives.")

    numeric_cols = get_numeric_columns(df)  # Get all numeric columns
    if not numeric_cols:  # No numeric columns available
        st.error("No numeric columns found in the dataset.")  # Show error
        return  # Exit

    col = st.selectbox("Select numeric column:", numeric_cols, key="norm_test_col")  # Column dropdown
    alpha = st.selectbox("Significance level (α):", [0.05, 0.01, 0.10], key="norm_test_alpha")  # Alpha

    if st.button("Run Normality Test"):  # Only run when button clicked
        series = drop_missing(df[col])  # Remove NaN values
        n = len(series)  # Sample size

        if n < 3:  # Need at least 3 observations
            st.error("Need at least 3 data points for a normality test.")  # Error
            return  # Exit

        # ---- Shapiro-Wilk test ----
        # Reliable for n up to 5000. For larger samples, we sample 5000 points.
        if n <= 5000:  # Sample size is fine for Shapiro
            shapiro_data = series  # Use all data
            shapiro_note = ""  # No note needed
        else:  # Sample too large for Shapiro
            shapiro_data = series.sample(5000, random_state=42)  # Sample 5000 points
            shapiro_note = " (computed on a random sample of 5,000 points)"  # Explain

        sw_stat, sw_p = sp.shapiro(shapiro_data)  # Run Shapiro-Wilk test

        # ---- D'Agostino-Pearson test ----
        # Combines skewness and kurtosis into one normality test. Needs n >= 8.
        if n >= 8:  # Enough data for D'Agostino
            da_stat, da_p = sp.normaltest(series)  # Run D'Agostino-Pearson test
        else:  # Too few points
            da_stat, da_p = None, None  # Cannot compute

        # ---- Compute skewness and kurtosis for interpretation ----
        skewness = float(sp.skew(series))  # Asymmetry measure (0 = symmetric)
        kurtosis = float(sp.kurtosis(series))  # Tail heaviness (0 = normal tails)

        # ---- Display results ----
        with st.expander(f"Normality Test Results for '{col}'", expanded=True):  # Results box
            st.markdown(f"**H₀:** '{col}' IS normally distributed")  # Null hypothesis
            st.markdown(f"**H₁:** '{col}' is NOT normally distributed")  # Alternative
            st.markdown("---")  # Divider

            # Show the two test results as metrics
            c1, c2, c3 = st.columns(3)  # Three metric columns
            with c1:  # First column
                st.metric("Shapiro-Wilk W", f"{sw_stat:.4f}")  # W statistic
                st.caption(f"p = {format_pvalue(sw_p)}{shapiro_note}")  # p-value with note
            with c2:  # Second column
                if da_p is not None:  # D'Agostino ran
                    st.metric("D'Agostino K²", f"{da_stat:.4f}")  # K² statistic
                    st.caption(f"p = {format_pvalue(da_p)}")  # p-value
                else:  # D'Agostino skipped
                    st.metric("D'Agostino K²", "N/A")  # Not available
                    st.caption("Needs n ≥ 8")  # Explain why
            with c3:  # Third column
                st.metric("Sample size (n)", str(n))  # Show n

            st.markdown("---")  # Divider

            # ---- Decision based on Shapiro-Wilk (primary test) ----
            st.markdown(f"**Significance Level (α):** {alpha}")  # Show alpha
            if sw_p > alpha:  # p > alpha → cannot reject normality
                st.success(  # Show a green success message
                    f"🟢 **Data appears NORMAL** (Shapiro-Wilk p = {format_pvalue(sw_p)} > α = {alpha})\n\n"
                    f"✅ You CAN use parametric tests: t-test, Z-test, ANOVA, Pearson correlation."
                )  # Green success message
            else:  # p <= alpha → reject normality
                st.warning(  # Show a yellow warning message
                    f"🔴 **Data is NOT normal** (Shapiro-Wilk p = {format_pvalue(sw_p)} ≤ α = {alpha})\n\n"
                    f"⚠️ Consider non-parametric tests instead: Mann-Whitney U, Wilcoxon, Kruskal-Wallis.\n\n"
                    f"Note: For large samples (n ≥ 30), parametric tests are still robust due to the "
                    f"Central Limit Theorem, even if the raw data is not normal."
                )  # Warning with guidance

            # ---- Show distribution shape statistics ----
            st.markdown("**Distribution Shape:**")  # Section label
            st.write(f"- Skewness: `{skewness:.4f}` "  # Skewness with interpretation
                     f"({'symmetric' if abs(skewness) < 0.5 else 'moderately skewed' if abs(skewness) < 1 else 'highly skewed'})")
            st.write(f"- Kurtosis: `{kurtosis:.4f}` "  # Kurtosis with interpretation
                     f"({'normal tails' if abs(kurtosis) < 0.5 else 'heavier/lighter tails than normal'})")

            # ---- Formula reference ----
            st.markdown("**About these tests:**")  # Section label
            st.write("- **Shapiro-Wilk**: Compares your data's order statistics to those expected "  # Display explanatory text in the UI
                     "from a normal distribution. W close to 1.0 means normal.")
            st.write("- **D'Agostino-Pearson**: Combines skewness and kurtosis. "  # Display explanatory text in the UI
                     "K² near 0 means normal shape.")

        # ---- Visual: histogram + Q-Q plot ----
        _plot_normality_visual(series, col)  # Draw the diagnostic plots


def _plot_normality_visual(series: pd.Series, col: str):
    """
    Draw a histogram with a fitted normal curve, plus a Q-Q plot.
    These visuals help confirm the numeric test results.

    Parameters:
        series (pd.Series): Cleaned numeric data
        col (str): Column name for labels
    """
    import matplotlib.pyplot as plt  # Import matplotlib for plotting

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))  # Create 2 side-by-side plots

    # ---- Left: histogram with fitted normal curve ----
    axes[0].hist(series, bins=30, density=True, alpha=0.6,  # Histogram normalized to density
                 color="#5B8FF9", edgecolor="white", label="Actual data")  # Blue bars
    mu, sigma = series.mean(), series.std()  # Estimate mean and std for the normal curve
    x = np.linspace(series.min(), series.max(), 300)  # X range for the curve
    axes[0].plot(x, sp.norm.pdf(x, mu, sigma), color="#E8684A", linewidth=2.5,  # Normal curve
                 label=f"Normal curve\nμ={mu:.2f}, σ={sigma:.2f}")  # Red line with params
    axes[0].set_title(f"Distribution of {col}", fontsize=12, fontweight="bold")  # Title
    axes[0].set_xlabel(label_with_unit(col))  # X label with unit
    axes[0].set_ylabel("Density")  # Y label
    axes[0].legend(fontsize=9)  # Legend
    axes[0].grid(alpha=0.3)  # Faint grid

    # ---- Right: Q-Q plot ----
    # Points on the line = normal. Points curving away = non-normal.
    (osm, osr), (slope, intercept, r) = sp.probplot(series, dist="norm")  # Compute Q-Q data
    axes[1].scatter(osm, osr, color="#5B8FF9", s=10, alpha=0.6, label="Data quantiles")  # Points
    axes[1].plot(osm, slope * np.array(osm) + intercept, color="#E8684A", linewidth=2,  # Reference line
                 label="Perfect normal line")  # Red line
    axes[1].set_title("Q-Q Plot (Normality Check)", fontsize=12, fontweight="bold")  # Title
    axes[1].set_xlabel("Theoretical Quantiles")  # X label
    axes[1].set_ylabel(f"Sample Quantiles of {label_with_unit(col)}")  # Y label
    axes[1].legend(fontsize=9)  # Legend
    axes[1].grid(alpha=0.3)  # Faint grid

    plt.tight_layout()  # Prevent label overlap
    st.pyplot(fig)  # Render in Streamlit
    plt.close(fig)  # Free memory


# =====================================================================
# MANN-WHITNEY U TEST (non-parametric two-sample)
# =====================================================================

def render_mann_whitney(df: pd.DataFrame):
    """
    Mann-Whitney U test — the non-parametric alternative to the two-sample t-test.
    Compares whether two independent groups have different distributions (medians).

    Use this instead of a t-test when the data is NOT normally distributed.

    H0: The two groups have the same distribution (same median)
    H1: The two groups have different distributions (different medians)
    """
    st.markdown("### Mann-Whitney U Test")  # Sub-heading
    st.write("Non-parametric alternative to the two-sample t-test. "  # Display explanatory text in the UI
             "Compares two independent groups WITHOUT assuming normality. "
             "Tests whether one group tends to have larger values than the other.")

    numeric_cols = get_numeric_columns(df)  # Numeric columns for the measured variable
    cat_cols = get_categorical_columns(df)  # Categorical columns for grouping

    if not numeric_cols:  # No numeric columns
        st.error("No numeric columns found.")  # Error
        return  # Exit
    if not cat_cols:  # No categorical columns to split groups
        st.error("Mann-Whitney U test requires a categorical column to define two groups.")  # Error
        return  # Exit

    num_col = st.selectbox("Numeric column to compare:", numeric_cols, key="mw_num")  # Measure
    group_col = st.selectbox("Grouping column:", cat_cols, key="mw_grp")  # Grouping variable

    unique_groups = df[group_col].dropna().unique().tolist()  # All unique group values
    if len(unique_groups) < 2:  # Need at least 2 groups
        st.warning(f"Column '{group_col}' has fewer than 2 groups.")  # Warn
        return  # Exit

    g1 = st.selectbox("Group 1:", unique_groups, key="mw_g1")  # First group
    g2 = st.selectbox("Group 2:", [g for g in unique_groups if g != g1], key="mw_g2")  # Second group
    alpha = st.selectbox("Significance level (α):", [0.05, 0.01, 0.10], key="mw_alpha")  # Alpha

    if st.button("Run Mann-Whitney U Test"):  # Run on click
        grp1 = drop_missing(df[df[group_col] == g1][num_col])  # Group 1 data
        grp2 = drop_missing(df[df[group_col] == g2][num_col])  # Group 2 data

        if len(grp1) < 1 or len(grp2) < 1:  # Each group needs data
            st.error("Each group needs at least 1 value.")  # Error
            return  # Exit

        # Run Mann-Whitney U test (two-sided)
        u_stat, p_value = sp.mannwhitneyu(grp1, grp2, alternative="two-sided")  # The test

        # Display results
        with st.expander("Mann-Whitney U Test Results", expanded=True):  # Results box
            st.markdown(f"**H₀:** Distribution of '{num_col}' is the same for {g1} and {g2}")  # H0
            st.markdown(f"**H₁:** The two groups have different distributions")  # H1
            st.markdown("---")  # Divider

            c1, c2, c3 = st.columns(3)  # Three metric columns
            with c1:  # First
                st.metric("U statistic", f"{u_stat:.1f}")  # U value
            with c2:  # Second
                st.metric("P-value", format_pvalue(p_value))  # p-value
            with c3:  # Third
                st.metric("Total n", str(len(grp1) + len(grp2)))  # Combined sample size

            st.markdown("---")  # Divider

            # Show median of each group (Mann-Whitney compares medians/ranks)
            st.markdown("**Group Medians (what this test compares):**")  # Label
            st.write(f"- {g1}: median = `{grp1.median():.4f}` (n = {len(grp1)})")  # Group 1 median
            st.write(f"- {g2}: median = `{grp2.median():.4f}` (n = {len(grp2)})")  # Group 2 median

            st.markdown(f"**Significance Level (α):** {alpha}")  # Alpha
            if p_value < alpha:  # Significant
                st.error(  # Show a red error / reject message
                    f"🔴 **Reject H₀** — p = {format_pvalue(p_value)} < α = {alpha}\n\n"
                    f"The two groups have significantly different distributions of '{num_col}'."
                )  # Reject message
            else:  # Not significant
                st.success(  # Show a green success message
                    f"🟢 **Fail to Reject H₀** — p = {format_pvalue(p_value)} ≥ α = {alpha}\n\n"
                    f"No significant difference between the two groups."
                )  # Fail to reject message

            # Formula / method note
            st.markdown("**How it works:** Mann-Whitney ranks all values from both groups together, "  # Render formatted text in the UI
                        "then checks whether one group's ranks are systematically higher. "
                        "Unlike the t-test, it does not assume normal data.")  # Explanation


# =====================================================================
# WILCOXON SIGNED-RANK TEST (non-parametric paired)
# =====================================================================

def render_wilcoxon(df: pd.DataFrame):
    """
    Wilcoxon Signed-Rank test — the non-parametric alternative to the paired t-test.
    Compares two RELATED measurements (e.g. before vs after) on the same subjects.

    Use this instead of a paired t-test when differences are NOT normally distributed.

    H0: The median difference between the pairs is zero
    H1: The median difference is not zero
    """
    st.markdown("### Wilcoxon Signed-Rank Test")  # Sub-heading
    st.write("Non-parametric alternative to the paired t-test. "  # Display explanatory text in the UI
             "Compares two RELATED numeric columns (e.g. before vs after, or two measurements "
             "on the same subjects) WITHOUT assuming normality.")

    numeric_cols = get_numeric_columns(df)  # Numeric columns
    if len(numeric_cols) < 2:  # Need 2 numeric columns to pair
        st.error("Wilcoxon Signed-Rank test requires at least 2 numeric columns "  # Show a red error / reject message
                 "(two paired measurements).")  # Error
        return  # Exit

    st.info("ℹ️ This test pairs the two columns row-by-row. Make sure the two columns "  # Show an informational message
            "represent related measurements on the SAME subjects (e.g. Score_Before and Score_After).")  # Guidance

    col1 = st.selectbox("First measurement column:", numeric_cols, key="wx_c1")  # First column
    col2 = st.selectbox("Second measurement column:",  # Second column
                        [c for c in numeric_cols if c != col1], key="wx_c2")  # Exclude first
    alpha = st.selectbox("Significance level (α):", [0.05, 0.01, 0.10], key="wx_alpha")  # Alpha

    if st.button("Run Wilcoxon Signed-Rank Test"):  # Run on click
        # Keep only rows where BOTH columns have values (paired data)
        paired = df[[col1, col2]].dropna()  # Drop rows missing either value

        if len(paired) < 6:  # Wilcoxon needs a reasonable number of pairs
            st.error(f"Need at least 6 complete pairs. After removing missing values, "  # Show a red error / reject message
                     f"only {len(paired)} pairs remain.")  # Error
            return  # Exit

        # Compute the differences between the paired measurements
        differences = paired[col1] - paired[col2]  # Difference per pair

        # Check if all differences are zero (test cannot run)
        if (differences == 0).all():  # All pairs identical
            st.error("All pairs are identical (all differences are zero). "  # Show a red error / reject message
                     "The test cannot be computed.")  # Error
            return  # Exit

        # Run Wilcoxon signed-rank test
        try:
            w_stat, p_value = sp.wilcoxon(paired[col1], paired[col2])  # The test
        except ValueError as e:  # Handle edge cases (e.g. too many zero differences)
            st.error(f"Wilcoxon test could not run: {e}")  # Show error
            return  # Exit

        # Display results
        with st.expander("Wilcoxon Signed-Rank Test Results", expanded=True):  # Results box
            st.markdown(f"**H₀:** Median difference between '{col1}' and '{col2}' is zero")  # H0
            st.markdown(f"**H₁:** The median difference is NOT zero")  # H1
            st.markdown("---")  # Divider

            c1, c2, c3 = st.columns(3)  # Three metric columns
            with c1:  # First
                st.metric("W statistic", f"{w_stat:.1f}")  # W value
            with c2:  # Second
                st.metric("P-value", format_pvalue(p_value))  # p-value
            with c3:  # Third
                st.metric("Number of pairs", str(len(paired)))  # Pair count

            st.markdown("---")  # Divider

            # Show summary of the differences
            st.markdown("**Summary of paired differences:**")  # Label
            st.write(f"- Median of '{col1}': `{paired[col1].median():.4f}`")  # Median 1
            st.write(f"- Median of '{col2}': `{paired[col2].median():.4f}`")  # Median 2
            st.write(f"- Median difference: `{differences.median():.4f}`")  # Median diff
            st.write(f"- Pairs where {col1} > {col2}: {(differences > 0).sum()}")  # Positive diffs
            st.write(f"- Pairs where {col1} < {col2}: {(differences < 0).sum()}")  # Negative diffs

            st.markdown(f"**Significance Level (α):** {alpha}")  # Alpha
            if p_value < alpha:  # Significant
                st.error(  # Show a red error / reject message
                    f"🔴 **Reject H₀** — p = {format_pvalue(p_value)} < α = {alpha}\n\n"
                    f"There is a significant difference between '{col1}' and '{col2}'."
                )  # Reject
            else:  # Not significant
                st.success(  # Show a green success message
                    f"🟢 **Fail to Reject H₀** — p = {format_pvalue(p_value)} ≥ α = {alpha}\n\n"
                    f"No significant difference between the two measurements."
                )  # Fail to reject

            st.markdown("**How it works:** Wilcoxon ranks the absolute differences between pairs, "  # Render formatted text in the UI
                        "then checks whether positive and negative differences balance out. "
                        "It is the non-parametric version of the paired t-test.")  # Explanation


# =====================================================================
# KRUSKAL-WALLIS TEST (non-parametric ANOVA)
# =====================================================================

def render_kruskal_wallis(df: pd.DataFrame):
    """
    Kruskal-Wallis H test — the non-parametric alternative to one-way ANOVA.
    Compares whether 3 or more independent groups have different distributions.

    Use this instead of ANOVA when the data is NOT normally distributed.

    H0: All groups have the same distribution (same median)
    H1: At least one group has a different distribution
    """
    st.markdown("### Kruskal-Wallis H Test")  # Sub-heading
    st.write("Non-parametric alternative to one-way ANOVA. "  # Display explanatory text in the UI
             "Compares 3 or more independent groups WITHOUT assuming normality. "
             "Tests whether at least one group tends to have different values.")

    numeric_cols = get_numeric_columns(df)  # Numeric columns
    cat_cols = get_categorical_columns(df)  # Categorical columns

    if not numeric_cols:  # No numeric columns
        st.error("No numeric columns found.")  # Error
        return  # Exit
    if not cat_cols:  # No grouping columns
        st.error("Kruskal-Wallis test requires a categorical grouping column.")  # Error
        return  # Exit

    num_col = st.selectbox("Numeric column:", numeric_cols, key="kw_num")  # Measure
    group_col = st.selectbox("Grouping column:", cat_cols, key="kw_grp")  # Grouping
    alpha = st.selectbox("Significance level (α):", [0.05, 0.01, 0.10], key="kw_alpha")  # Alpha

    if st.button("Run Kruskal-Wallis Test"):  # Run on click
        group_labels = df[group_col].dropna().unique()  # All unique groups
        groups = []  # List to hold each group's data
        valid_labels = []  # Labels of groups with enough data

        for g in group_labels:  # Loop over each group
            grp_data = drop_missing(df[df[group_col] == g][num_col])  # Group data
            if len(grp_data) >= 1:  # Group has data
                groups.append(grp_data)  # Add to list
                valid_labels.append(g)  # Record label

        if len(groups) < 2:  # Need at least 2 groups
            st.error("Need at least 2 groups with data.")  # Error
            return  # Exit

        # Run Kruskal-Wallis H test (* unpacks the list of groups)
        h_stat, p_value = sp.kruskal(*groups)  # The test
        df_val = len(groups) - 1  # Degrees of freedom = k - 1

        # Display results
        with st.expander("Kruskal-Wallis Test Results", expanded=True):  # Results box
            st.markdown(f"**H₀:** All groups have the same distribution of '{num_col}'")  # H0
            st.markdown(f"**H₁:** At least one group has a different distribution")  # H1
            st.markdown("---")  # Divider

            c1, c2, c3 = st.columns(3)  # Three metric columns
            with c1:  # First
                st.metric("H statistic", f"{h_stat:.4f}")  # H value
            with c2:  # Second
                st.metric("P-value", format_pvalue(p_value))  # p-value
            with c3:  # Third
                st.metric("Degrees of freedom", str(df_val))  # df = k-1

            st.markdown("---")  # Divider

            # Show median of each group as a table
            st.markdown("**Group Medians (what this test compares):**")  # Label
            median_df = pd.DataFrame({  # Build summary table
                "Group": [str(g) for g in valid_labels],  # Group names
                "Median": [round(grp.median(), 4) for grp in groups],  # Medians
                "n": [len(grp) for grp in groups],  # Counts
            })  # End DataFrame
            st.dataframe(median_df, use_container_width=True, hide_index=True)  # Show table

            st.markdown(f"**Significance Level (α):** {alpha}")  # Alpha
            if p_value < alpha:  # Significant
                st.error(  # Show a red error / reject message
                    f"🔴 **Reject H₀** — p = {format_pvalue(p_value)} < α = {alpha}\n\n"
                    f"At least one group has a significantly different distribution of '{num_col}'."
                )  # Reject
            else:  # Not significant
                st.success(  # Show a green success message
                    f"🟢 **Fail to Reject H₀** — p = {format_pvalue(p_value)} ≥ α = {alpha}\n\n"
                    f"No significant difference between the groups."
                )  # Fail to reject

            st.markdown("**How it works:** Kruskal-Wallis ranks all values across all groups, "  # Render formatted text in the UI
                        "then checks whether the average rank differs between groups. "
                        "It is the non-parametric version of one-way ANOVA.")  # Explanation


# =====================================================================
# FISHER'S EXACT TEST (2x2 categorical)
# =====================================================================

def render_fishers_exact(df: pd.DataFrame):
    """
    Fisher's Exact Test — tests association between two categorical variables.
    It is the preferred alternative to chi-square when the table is 2x2 and/or
    sample sizes are small (expected cell counts < 5).

    Unlike chi-square (which is an approximation), Fisher's test gives an EXACT p-value.

    H0: The two variables are independent (no association)
    H1: The two variables are associated
    """
    st.markdown("### Fisher's Exact Test")  # Sub-heading
    st.write("Tests association between two categorical variables in a 2x2 table. "  # Display explanatory text in the UI
             "Preferred over chi-square when sample sizes are small (expected counts < 5). "
             "Gives an EXACT p-value rather than an approximation.")

    cat_cols = get_categorical_columns(df)  # Categorical columns

    if len(cat_cols) < 2:  # Need 2 categorical columns
        st.error("Fisher's Exact Test requires at least 2 categorical columns.")  # Error
        return  # Exit

    var1 = st.selectbox("Variable 1:", cat_cols, key="fisher_v1")  # First variable
    var2 = st.selectbox("Variable 2:", [c for c in cat_cols if c != var1], key="fisher_v2")  # Second
    alpha = st.selectbox("Significance level (α):", [0.05, 0.01, 0.10], key="fisher_alpha")  # Alpha

    if st.button("Run Fisher's Exact Test"):  # Run on click
        clean_df = df[[var1, var2]].dropna()  # Drop rows missing either value

        if len(clean_df) < 2:  # Need some data
            st.error("Not enough complete observations.")  # Error
            return  # Exit

        # Build a contingency table
        contingency = pd.crosstab(clean_df[var1], clean_df[var2])  # Cross-tabulation

        st.markdown("**Contingency Table (Observed Frequencies):**")  # Label
        st.dataframe(contingency, use_container_width=True)  # Show observed counts

        # Fisher's exact test in scipy only works on 2x2 tables
        if contingency.shape != (2, 2):  # Table is not 2x2
            st.error(f"""  # Show a red error / reject message
            ❌ **Fisher's Exact Test requires a 2x2 table.**

            Your table is {contingency.shape[0]}x{contingency.shape[1]} because:
            - '{var1}' has {contingency.shape[0]} categories
            - '{var2}' has {contingency.shape[1]} categories

            **What to do:**
            - Choose two variables that each have exactly 2 categories (e.g. Yes/No, Male/Female), OR
            - Use the **Chi-Square Test** instead, which handles larger tables.
            """)  # Detailed explanation of why it can't run
            return  # Exit

        # Run Fisher's exact test on the 2x2 table
        odds_ratio, p_value = sp.fisher_exact(contingency)  # The test

        # Display results
        with st.expander("Fisher's Exact Test Results", expanded=True):  # Results box
            st.markdown(f"**H₀:** '{var1}' and '{var2}' are independent")  # H0
            st.markdown(f"**H₁:** '{var1}' and '{var2}' are associated")  # H1
            st.markdown("---")  # Divider

            c1, c2, c3 = st.columns(3)  # Three metric columns
            with c1:  # First
                st.metric("Odds Ratio", f"{odds_ratio:.4f}")  # Odds ratio
            with c2:  # Second
                st.metric("P-value (exact)", format_pvalue(p_value))  # Exact p-value
            with c3:  # Third
                st.metric("Total n", str(len(clean_df)))  # Sample size

            st.markdown("---")  # Divider

            st.markdown(f"**Significance Level (α):** {alpha}")  # Alpha
            if p_value < alpha:  # Significant
                st.error(
                    f"🔴 **Reject H₀** — p = {format_pvalue(p_value)} < α = {alpha}\n\n"
                    f"'{var1}' and '{var2}' are significantly associated."
                )  # Reject
            else:  # Not significant
                st.success(
                    f"🟢 **Fail to Reject H₀** — p = {format_pvalue(p_value)} ≥ α = {alpha}\n\n"
                    f"No significant association between '{var1}' and '{var2}'."
                )  # Fail to reject

            # Interpret the odds ratio
            st.markdown("**Odds Ratio interpretation:**")  # Label
            if odds_ratio > 1:  # OR > 1
                st.write(f"- An odds ratio of {odds_ratio:.2f} means the odds of the outcome "
                         f"are {odds_ratio:.2f}x higher in one group than the other.")  # Explain
            elif odds_ratio < 1:  # OR < 1
                st.write(f"- An odds ratio of {odds_ratio:.2f} means the odds are lower in one group.")  # Explain
            else:  # OR = 1
                st.write("- An odds ratio of 1.0 means no difference in odds between groups.")  # Explain

            st.markdown("**Why Fisher's instead of chi-square?** Fisher's test computes the EXACT "
                        "probability of the observed table, making it reliable even for small samples "
                        "where chi-square's approximation breaks down.")  # Explanation


# =====================================================================
# TEST ADVISOR — recommends which test to use based on the data
# =====================================================================

def render_test_advisor(df: pd.DataFrame):
    """
    Interactive Test Advisor.
    The user describes their goal and selects their variables.
    The advisor examines the real data (sample size, normality, number of groups)
    and recommends the most appropriate statistical test, with reasons.
    """
    st.markdown("### 🧭 Statistical Test Advisor")  # Sub-heading
    st.write("Not sure which test to use? Answer a few questions about your data, "
             "and this advisor will examine your dataset and recommend the right test.")

    numeric_cols = get_numeric_columns(df)  # Numeric columns
    cat_cols = get_categorical_columns(df)  # Categorical columns

    # ---- Step 1: What is the user's goal? ----
    st.markdown("#### Step 1: What do you want to do?")  # Step heading
    goal = st.radio(  # Radio button for the analysis goal
        "Choose your analysis goal:",
        [
            "Compare a group's average to a known value",       # → one-sample test
            "Compare averages between TWO groups",              # → two-sample test
            "Compare averages between THREE or more groups",    # → ANOVA family
            "Compare TWO related measurements (before/after)",  # → paired test
            "Check the relationship between TWO numeric variables",  # → correlation
            "Check association between TWO categorical variables",   # → chi-square/Fisher
        ],
        key="advisor_goal"
    )

    st.markdown("---")  # Divider

    # ---- Step 2: Select the relevant variables and run the recommendation ----
    st.markdown("#### Step 2: Select your variables")  # Step heading

    # ====== GOAL 1: one-sample ======
    if goal == "Compare a group's average to a known value":  # One-sample scenario
        if not numeric_cols:  # Need numeric data
            st.error("You need at least one numeric column for this analysis.")  # Error
            return  # Exit
        col = st.selectbox("Which numeric variable?", numeric_cols, key="adv1_col")  # Variable
        if st.button("Get Recommendation", key="adv1_btn"):  # Run on click
            series = drop_missing(df[col])  # Clean data
            n = len(series)  # Sample size
            is_normal, sw_p = _check_normal(series)  # Check normality
            _advisor_result(  # Show recommendation
                primary="One-Sample t-test" if (is_normal or n >= 30) else "One-Sample t-test (with caution)",
                reason=_build_reason(n, is_normal, sw_p, groups=1),  # Build the reasoning text
                alternative="Wilcoxon Signed-Rank test (against a hypothesized median)" if not is_normal and n < 30 else None,
                where="Hypothesis Testing → One-Sample T-test",  # Where to find it
            )

    # ====== GOAL 2: two independent groups ======
    elif goal == "Compare averages between TWO groups":  # Two-sample scenario
        if not numeric_cols or not cat_cols:  # Need both types
            st.error("You need one numeric column AND one categorical column (to define the two groups).")  # Error
            return  # Exit
        num_col = st.selectbox("Numeric variable to compare:", numeric_cols, key="adv2_num")  # Measure
        grp_col = st.selectbox("Grouping variable:", cat_cols, key="adv2_grp")  # Groups
        if st.button("Get Recommendation", key="adv2_btn"):  # Run on click
            n_groups = df[grp_col].dropna().nunique()  # Number of groups
            if n_groups != 2:  # Warn if not exactly 2
                st.warning(f"⚠️ '{grp_col}' has {n_groups} groups, not 2. "
                           f"For exactly 2 groups use a t-test/Mann-Whitney; for 3+ see the ANOVA option.")  # Warn
            series = drop_missing(df[num_col])  # Clean data
            n = len(series)  # Sample size
            is_normal, sw_p = _check_normal(series)  # Normality
            if is_normal or n >= 30:  # Parametric OK
                _advisor_result(
                    primary="Two-Sample (Independent) t-test",  # Recommended test
                    reason=_build_reason(n, is_normal, sw_p, groups=2),  # Reasoning
                    alternative="Mann-Whitney U test (if you prefer not to assume normality)",  # Alternative
                    where="Hypothesis Testing → Two-Sample T-test",  # Location
                )
            else:  # Non-parametric recommended
                _advisor_result(
                    primary="Mann-Whitney U test",  # Recommended (non-parametric)
                    reason=_build_reason(n, is_normal, sw_p, groups=2),  # Reasoning
                    alternative="Two-Sample t-test (only if you can justify normality)",  # Alternative
                    where="Non-Parametric Tests → Mann-Whitney U Test",  # Location
                )

    # ====== GOAL 3: three or more groups ======
    elif goal == "Compare averages between THREE or more groups":  # ANOVA scenario
        if not numeric_cols or not cat_cols:  # Need both types
            st.error("You need one numeric column AND one categorical column with 3+ groups.")  # Error
            return  # Exit
        num_col = st.selectbox("Numeric variable to compare:", numeric_cols, key="adv3_num")  # Measure
        grp_col = st.selectbox("Grouping variable:", cat_cols, key="adv3_grp")  # Groups
        if st.button("Get Recommendation", key="adv3_btn"):  # Run on click
            n_groups = df[grp_col].dropna().nunique()  # Number of groups
            series = drop_missing(df[num_col])  # Clean data
            n = len(series)  # Sample size
            is_normal, sw_p = _check_normal(series)  # Normality
            st.write(f"'{grp_col}' has **{n_groups} groups**.")  # Report group count
            if is_normal or n >= 30:  # Parametric OK
                _advisor_result(
                    primary="One-Way ANOVA",  # Recommended
                    reason=_build_reason(n, is_normal, sw_p, groups=n_groups),  # Reasoning
                    alternative="Kruskal-Wallis test (if you prefer not to assume normality)",  # Alternative
                    where="Hypothesis Testing → One-Way ANOVA",  # Location
                )
            else:  # Non-parametric recommended
                _advisor_result(
                    primary="Kruskal-Wallis H test",  # Recommended (non-parametric)
                    reason=_build_reason(n, is_normal, sw_p, groups=n_groups),  # Reasoning
                    alternative="One-Way ANOVA (only if normality can be justified)",  # Alternative
                    where="Non-Parametric Tests → Kruskal-Wallis Test",  # Location
                )

    # ====== GOAL 4: paired measurements ======
    elif goal == "Compare TWO related measurements (before/after)":  # Paired scenario
        if len(numeric_cols) < 2:  # Need 2 numeric columns
            st.error("You need at least 2 numeric columns (the two related measurements).")  # Error
            return  # Exit
        c1 = st.selectbox("First measurement:", numeric_cols, key="adv4_c1")  # First
        c2 = st.selectbox("Second measurement:", [c for c in numeric_cols if c != c1], key="adv4_c2")  # Second
        if st.button("Get Recommendation", key="adv4_btn"):  # Run on click
            paired = df[[c1, c2]].dropna()  # Paired data
            diffs = paired[c1] - paired[c2]  # Differences
            n = len(diffs)  # Number of pairs
            is_normal, sw_p = _check_normal(diffs)  # Normality of DIFFERENCES (key for paired)
            st.write(f"Number of complete pairs: **{n}**")  # Report pairs
            if is_normal or n >= 30:  # Parametric OK
                _advisor_result(
                    primary="Paired t-test",  # Recommended
                    reason=f"The differences between pairs are {'approximately normal' if is_normal else 'non-normal, but n≥30 so the CLT applies'} "
                           f"(Shapiro p = {format_pvalue(sw_p)}, n = {n} pairs).",  # Reasoning about differences
                    alternative="Wilcoxon Signed-Rank test",  # Alternative
                    where="(Paired t-test not in current menu — use Wilcoxon below, or compute differences manually)",  # Location note
                )
            else:  # Non-parametric recommended
                _advisor_result(
                    primary="Wilcoxon Signed-Rank test",  # Recommended (non-parametric)
                    reason=f"The differences between pairs are NOT normally distributed "
                           f"(Shapiro p = {format_pvalue(sw_p)}) and there are only {n} pairs (< 30), "
                           f"so a non-parametric test is safer.",  # Reasoning
                    alternative="Paired t-test (only if normality can be justified)",  # Alternative
                    where="Non-Parametric Tests → Wilcoxon Signed-Rank Test",  # Location
                )

    # ====== GOAL 5: correlation ======
    elif goal == "Check the relationship between TWO numeric variables":  # Correlation scenario
        if len(numeric_cols) < 2:  # Need 2 numeric columns
            st.error("You need at least 2 numeric columns.")  # Error
            return  # Exit
        c1 = st.selectbox("First numeric variable:", numeric_cols, key="adv5_c1")  # First
        c2 = st.selectbox("Second numeric variable:", [c for c in numeric_cols if c != c1], key="adv5_c2")  # Second
        if st.button("Get Recommendation", key="adv5_btn"):  # Run on click
            pair = df[[c1, c2]].dropna()  # Paired data
            n = len(pair)  # Sample size
            norm1, p1 = _check_normal(pair[c1])  # Normality of variable 1
            norm2, p2 = _check_normal(pair[c2])  # Normality of variable 2
            both_normal = norm1 and norm2  # Both must be normal for Pearson
            if both_normal or n >= 30:  # Parametric OK
                _advisor_result(
                    primary="Pearson Correlation",  # Recommended
                    reason=f"Both variables are {'approximately normal' if both_normal else 'non-normal, but n≥30 so Pearson is robust'} "
                           f"(Shapiro p: {format_pvalue(p1)} and {format_pvalue(p2)}, n = {n}). "
                           f"Pearson measures LINEAR relationships.",  # Reasoning
                    alternative="Spearman rank correlation (if the relationship is non-linear or monotonic)",  # Alternative
                    where="Hypothesis Testing → Pearson Correlation",  # Location
                )
            else:  # Non-parametric recommended
                _advisor_result(
                    primary="Spearman Rank Correlation",  # Recommended (non-parametric)
                    reason=f"At least one variable is not normally distributed "
                           f"(Shapiro p: {format_pvalue(p1)} and {format_pvalue(p2)}) with n = {n} < 30. "
                           f"Spearman works on ranks and handles non-linear monotonic relationships.",  # Reasoning
                    alternative="Pearson correlation (only if normality can be justified)",  # Alternative
                    where="(Use Pearson Correlation in Hypothesis Testing — it reports both, or interpret with caution)",  # Location note
                )

    # ====== GOAL 6: categorical association ======
    elif goal == "Check association between TWO categorical variables":  # Categorical scenario
        if len(cat_cols) < 2:  # Need 2 categorical columns
            st.error("You need at least 2 categorical columns.")  # Error
            return  # Exit
        v1 = st.selectbox("First categorical variable:", cat_cols, key="adv6_v1")  # First
        v2 = st.selectbox("Second categorical variable:", [c for c in cat_cols if c != v1], key="adv6_v2")  # Second
        if st.button("Get Recommendation", key="adv6_btn"):  # Run on click
            clean = df[[v1, v2]].dropna()  # Clean data
            ct = pd.crosstab(clean[v1], clean[v2])  # Contingency table
            _, _, _, expected = sp.chi2_contingency(ct)  # Get expected frequencies
            low_expected = (expected < 5).sum()  # Count cells with expected < 5
            is_2x2 = ct.shape == (2, 2)  # Is it a 2x2 table?
            st.write(f"Table size: **{ct.shape[0]}x{ct.shape[1]}** | "
                     f"Cells with expected count < 5: **{low_expected}**")  # Report table info
            if is_2x2 and low_expected > 0:  # 2x2 with small counts → Fisher
                _advisor_result(
                    primary="Fisher's Exact Test",  # Recommended
                    reason=f"Your table is 2x2 and {low_expected} cell(s) have expected counts < 5. "
                           f"Chi-square's approximation is unreliable here — Fisher's test gives an exact p-value.",  # Reasoning
                    alternative="Chi-square (not recommended due to small expected counts)",  # Alternative
                    where="Non-Parametric Tests → Fisher's Exact Test",  # Location
                )
            elif low_expected > 0:  # Larger table with small counts
                _advisor_result(
                    primary="Chi-square (with caution) or combine categories",  # Recommended
                    reason=f"Your table is {ct.shape[0]}x{ct.shape[1]} with {low_expected} low-count cell(s). "
                           f"Fisher's exact test in this tool only supports 2x2 tables. "
                           f"Consider merging rare categories, then use chi-square.",  # Reasoning
                    alternative="Fisher's Exact Test (only after reducing to 2x2)",  # Alternative
                    where="Hypothesis Testing → Chi-Square Test",  # Location
                )
            else:  # Counts are fine → chi-square
                _advisor_result(
                    primary="Chi-Square Test of Independence",  # Recommended
                    reason=f"Your table is {ct.shape[0]}x{ct.shape[1]} and all expected counts are ≥ 5. "
                           f"Chi-square is the standard, appropriate choice here.",  # Reasoning
                    alternative="Fisher's Exact Test (if you reduce to a 2x2 table)" if is_2x2 else None,  # Alternative
                    where="Hypothesis Testing → Chi-Square Test",  # Location
                )


def _check_normal(series: pd.Series):
    """
    Helper: check whether a series is approximately normal using Shapiro-Wilk.
    Returns (is_normal: bool, p_value: float).

    Parameters:
        series (pd.Series): Numeric data to test

    Returns:
        tuple: (True if normal at alpha=0.05, the Shapiro p-value)
    """
    s = drop_missing(series)  # Remove NaN
    if len(s) < 3:  # Too few points to test
        return False, 0.0  # Treat as non-normal
    if len(s) > 5000:  # Shapiro limit
        s = s.sample(5000, random_state=42)  # Sample down to 5000
    try:
        _, p = sp.shapiro(s)  # Run Shapiro-Wilk
    except Exception:  # Any error
        return False, 0.0  # Treat as non-normal
    return p > 0.05, p  # Normal if p > 0.05


def _build_reason(n, is_normal, sw_p, groups):
    """
    Helper: build a human-readable explanation for the test recommendation.

    Parameters:
        n (int): Sample size
        is_normal (bool): Whether data passed the normality test
        sw_p (float): Shapiro-Wilk p-value
        groups (int): Number of groups involved

    Returns:
        str: A sentence explaining the recommendation
    """
    if is_normal:  # Data is normal
        return (f"Your data appears normally distributed (Shapiro-Wilk p = {format_pvalue(sw_p)} > 0.05) "
                f"with n = {n}. Parametric tests are appropriate and most powerful here.")  # Reason
    elif n >= 30:  # Not normal but large sample
        return (f"Your data is NOT normally distributed (Shapiro-Wilk p = {format_pvalue(sw_p)} ≤ 0.05), "
                f"but with n = {n} (≥ 30) the Central Limit Theorem makes parametric tests robust.")  # Reason
    else:  # Not normal and small sample
        return (f"Your data is NOT normally distributed (Shapiro-Wilk p = {format_pvalue(sw_p)} ≤ 0.05) "
                f"and the sample is small (n = {n} < 30). A non-parametric test is safer.")  # Reason


def _advisor_result(primary, reason, alternative, where):
    """
    Helper: display the advisor's recommendation in a clear, styled box.

    Parameters:
        primary (str): The recommended test name
        reason (str): Why this test is recommended
        alternative (str or None): An alternative test, if any
        where (str): Where to find the test in the app
    """
    st.markdown("---")  # Divider
    st.markdown("#### ✅ Recommendation")  # Recommendation heading
    st.success(f"**Use: {primary}**")  # Highlight the recommended test
    st.markdown(f"**Why:** {reason}")  # Explain the reasoning
    if alternative:  # If there's an alternative
        st.info(f"**Alternative:** {alternative}")  # Show the alternative
    st.markdown(f"**Where to find it:** `{where}`")  # Tell user where to go
