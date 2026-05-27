# =========================
# modules/tests.py
# Hypothesis Testing module
# Fixes applied:
#   1. Two-sample T-test: added numeric-only split option (threshold)
#   2. One-way ANOVA: added Tukey HSD post-hoc test
#   3. Z-test: default mu0 is now the sample mean (not 0)
# =========================

import pandas as pd
import numpy as np
import scipy.stats as sp
import streamlit as st
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from utils.helpers import (
    get_numeric_columns,
    get_categorical_columns,
    drop_missing,
    section_header,
    format_pvalue,
)


def render_tests(df: pd.DataFrame):
    section_header("Hypothesis Testing", "🧪")

    test_type = st.selectbox(
        "Select hypothesis test:",
        [
            "One-Sample T-test",
            "Two-Sample T-test",
            "Z-test (One-Sample)",
            "One-Way ANOVA",
            "Two-Way ANOVA",
            "Chi-Square Test",
        ],
        key="test_type_select"
    )

    if test_type == "One-Sample T-test":
        _one_sample_ttest(df)
    elif test_type == "Two-Sample T-test":
        _two_sample_ttest(df)
    elif test_type == "Z-test (One-Sample)":
        _ztest_one_sample(df)
    elif test_type == "One-Way ANOVA":
        _one_way_anova(df)
    elif test_type == "Two-Way ANOVA":
        _two_way_anova(df)
    elif test_type == "Chi-Square Test":
        _chi_square_test(df)


def _one_sample_ttest(df: pd.DataFrame):
    """
    One-sample T-test.
    H0: mean of selected column = mu0
    H1: mean != mu0
    """
    st.markdown("### One-Sample T-test")
    st.write("Tests whether the sample mean equals a hypothesized population mean.")

    numeric_cols = get_numeric_columns(df)
    if not numeric_cols:
        st.error("No numeric columns available.")
        return

    col = st.selectbox("Select variable:", numeric_cols, key="t1_col")
    series = drop_missing(df[col])

    # FIX: default mu0 = sample mean so user sees a meaningful starting point
    sample_mean = float(round(series.mean(), 4))
    mu0 = st.number_input(
        "Hypothesized mean (μ₀):",
        value=sample_mean,
        step=float(max(0.01, sample_mean * 0.01)),
        key="t1_mu0",
        help=f"Default = sample mean ({sample_mean}). Change to test against a different value."
    )
    st.caption(f"Sample mean = {sample_mean} — change μ₀ to test a different hypothesis.")

    alpha = st.selectbox("Significance level (α):", [0.05, 0.01, 0.10], key="t1_alpha")

    if st.button("Run One-Sample T-test"):
        if len(series) < 2:
            st.error("Need at least 2 data points.")
            return

        t_stat, p_value = sp.ttest_1samp(series, popmean=mu0)
        df_val = len(series) - 1

        _display_test_results(
            test_name="One-Sample T-test",
            statistic_name="t-statistic",
            statistic=t_stat,
            p_value=p_value,
            df=df_val,
            alpha=alpha,
            h0=f"H₀: mean of '{col}' = {mu0}",
            h1=f"H₁: mean of '{col}' ≠ {mu0}",
            extra_info={
                "Sample Mean": round(series.mean(), 4),
                "Hypothesized Mean (μ₀)": mu0,
                "Difference (x̄ - μ₀)": round(series.mean() - mu0, 4),
                "Sample Std Dev": round(series.std(ddof=1), 4),
                "Sample Size (n)": len(series),
            }
        )
        _check_normality(series, col)


def _two_sample_ttest(df: pd.DataFrame):
    """
    Two-sample independent T-test.
    H0: mu1 = mu2
    H1: mu1 != mu2

    FIX: Added Option B — split a numeric column by threshold when no categorical exists.
    """
    st.markdown("### Two-Sample T-test")
    st.write("Tests whether the means of two independent groups are equal.")

    numeric_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)

    if not numeric_cols:
        st.error("No numeric columns available.")
        return

    # Choose split method
    if cat_cols:
        split_method = st.radio(
            "How to define the two groups?",
            ["Option A: Split by categorical column", "Option B: Split numeric column by threshold"],
            key="t2_split_method"
        )
    else:
        # No categorical columns — force Option B
        st.info("ℹ️ No categorical columns found. Using numeric threshold split (Option B).")
        split_method = "Option B: Split numeric column by threshold"

    num_col = st.selectbox("Numeric column to test:", numeric_cols, key="t2_num")

    if split_method == "Option A: Split by categorical column":
        # ------- OPTION A: group by categorical -------
        group_col = st.selectbox("Group column:", cat_cols, key="t2_grp")
        unique_groups = df[group_col].dropna().unique().tolist()

        if len(unique_groups) < 2:
            st.warning("Need at least 2 groups in the selected column.")
            return

        g1_label = st.selectbox("Group 1:", unique_groups, key="t2_g1")
        g2_label = st.selectbox("Group 2:", [g for g in unique_groups if g != g1_label], key="t2_g2")
        alpha = st.selectbox("Significance level (α):", [0.05, 0.01, 0.10], key="t2_alpha_a")

        if st.button("Run Two-Sample T-test"):
            grp1 = drop_missing(df[df[group_col] == g1_label][num_col])
            grp2 = drop_missing(df[df[group_col] == g2_label][num_col])

            if len(grp1) < 2 or len(grp2) < 2:
                st.error("Each group needs at least 2 values.")
                return

            _, levene_p = sp.levene(grp1, grp2)
            equal_var = levene_p > 0.05
            t_stat, p_value = sp.ttest_ind(grp1, grp2, equal_var=equal_var)
            df_val = len(grp1) + len(grp2) - 2

            _display_test_results(
                test_name="Two-Sample T-test",
                statistic_name="t-statistic",
                statistic=t_stat,
                p_value=p_value,
                df=df_val,
                alpha=alpha,
                h0=f"H₀: mean({g1_label}) = mean({g2_label}) for '{num_col}'",
                h1=f"H₁: mean({g1_label}) ≠ mean({g2_label}) for '{num_col}'",
                extra_info={
                    f"Mean — {g1_label}": round(grp1.mean(), 4),
                    f"Mean — {g2_label}": round(grp2.mean(), 4),
                    f"n₁ ({g1_label})": len(grp1),
                    f"n₂ ({g2_label})": len(grp2),
                    "Levene test (equal var)": f"p={levene_p:.4f} → {'Equal (Student t)' if equal_var else 'Unequal (Welch t)'}",
                }
            )
            _check_normality(grp1, g1_label)
            _check_normality(grp2, g2_label)

    else:
        # ------- OPTION B: split by numeric threshold -------
        series = drop_missing(df[num_col])

        st.info(f"""
        **Option B — Threshold split:**
        Rows where '{num_col}' > threshold → Group 1 (High)
        Rows where '{num_col}' ≤ threshold → Group 2 (Low)
        """)

        # Use median as default threshold — natural split point
        default_thresh = float(round(series.median(), 2))
        threshold = st.number_input(
            f"Threshold for '{num_col}':",
            value=default_thresh,
            step=float(max(0.01, series.std() * 0.1)),
            key="t2_threshold",
            help=f"Default = median ({default_thresh}). Splits data into roughly equal halves."
        )
        st.caption(f"Median = {default_thresh} | Mean = {series.mean():.2f}")

        # Choose which numeric column to compare between the two groups
        compare_col_options = [c for c in numeric_cols if c != num_col]
        if not compare_col_options:
            compare_col_options = numeric_cols  # fallback: same column

        compare_col = st.selectbox("Numeric column to compare:", compare_col_options, key="t2_compare_col",
                                   help="The variable whose mean you want to compare between the two groups.")
        alpha = st.selectbox("Significance level (α):", [0.05, 0.01, 0.10], key="t2_alpha_b")

        if st.button("Run Two-Sample T-test"):
            mask_high = df[num_col] > threshold
            grp1 = drop_missing(df.loc[mask_high, compare_col])
            grp2 = drop_missing(df.loc[~mask_high, compare_col])

            if len(grp1) < 2 or len(grp2) < 2:
                st.error(f"Each group needs ≥2 values. Got: Group1={len(grp1)}, Group2={len(grp2)}. Try a different threshold.")
                return

            _, levene_p = sp.levene(grp1, grp2)
            equal_var = levene_p > 0.05
            t_stat, p_value = sp.ttest_ind(grp1, grp2, equal_var=equal_var)
            df_val = len(grp1) + len(grp2) - 2

            _display_test_results(
                test_name="Two-Sample T-test (threshold split)",
                statistic_name="t-statistic",
                statistic=t_stat,
                p_value=p_value,
                df=df_val,
                alpha=alpha,
                h0=f"H₀: mean({num_col}>{threshold}) = mean({num_col}≤{threshold}) for '{compare_col}'",
                h1=f"H₁: mean({num_col}>{threshold}) ≠ mean({num_col}≤{threshold}) for '{compare_col}'",
                extra_info={
                    f"Group 1 ({num_col} > {threshold}) mean": round(grp1.mean(), 4),
                    f"Group 2 ({num_col} ≤ {threshold}) mean": round(grp2.mean(), 4),
                    "n₁ (High group)": len(grp1),
                    "n₂ (Low group)": len(grp2),
                    "Levene test (equal var)": f"p={levene_p:.4f} → {'Equal (Student t)' if equal_var else 'Unequal (Welch t)'}",
                }
            )


def _ztest_one_sample(df: pd.DataFrame):
    """
    One-sample Z-test.
    H0: mu = mu0    H1: mu != mu0
    FIX: default mu0 = sample mean instead of 0.
    """
    st.markdown("### Z-test (One-Sample)")
    st.write("Tests whether the sample mean equals a hypothesized value (best for n ≥ 30).")

    numeric_cols = get_numeric_columns(df)
    if not numeric_cols:
        st.error("No numeric columns available.")
        return

    col = st.selectbox("Select variable:", numeric_cols, key="z_col")
    series = drop_missing(df[col])

    # FIX: default mu0 = sample mean so user sees a meaningful starting point
    sample_mean = float(round(series.mean(), 4))
    mu0 = st.number_input(
        "Hypothesized mean (μ₀):",
        value=sample_mean,
        step=float(max(0.01, sample_mean * 0.01)),
        key="z_mu0",
        help=f"Default = sample mean ({sample_mean}). Change this to test against a specific value."
    )
    st.caption(f"ℹ️ Sample mean = **{sample_mean}**. Change μ₀ to a different value to test a real hypothesis (e.g. industry average).")

    use_known_sigma = st.checkbox("Use known population σ?", value=False, key="z_known_sig")
    if use_known_sigma:
        sigma = st.number_input("Population σ:", value=float(round(series.std(ddof=1), 4)),
                                min_value=0.001, key="z_sigma")
    else:
        sigma = None

    alpha = st.selectbox("Significance level (α):", [0.05, 0.01, 0.10], key="z_alpha")

    if st.button("Run Z-test"):
        if len(series) < 2:
            st.error("Need at least 2 data points.")
            return

        if len(series) < 30:
            st.warning(f"⚠️ n={len(series)} < 30. Z-test is designed for large samples. Consider T-test instead.")

        std_used = sigma if use_known_sigma else series.std(ddof=1)
        n = len(series)
        z_stat = (series.mean() - mu0) / (std_used / np.sqrt(n))
        p_value = 2 * (1 - sp.norm.cdf(abs(z_stat)))

        _display_test_results(
            test_name="One-Sample Z-test",
            statistic_name="z-statistic",
            statistic=z_stat,
            p_value=p_value,
            df=None,
            alpha=alpha,
            h0=f"H₀: mean of '{col}' = {mu0}",
            h1=f"H₁: mean of '{col}' ≠ {mu0}",
            extra_info={
                "Sample Mean (x̄)": round(series.mean(), 4),
                "Hypothesized Mean (μ₀)": mu0,
                "Difference (x̄ - μ₀)": round(series.mean() - mu0, 4),
                "σ used": round(std_used, 4),
                "Source of σ": "Known population σ" if use_known_sigma else "Sample std dev (approximation)",
                "n": n,
                "Z formula": f"z = ({series.mean():.2f} - {mu0}) / ({std_used:.2f} / √{n}) = {z_stat:.4f}",
            }
        )


def _one_way_anova(df: pd.DataFrame):
    """
    One-Way ANOVA.
    H0: all group means are equal
    H1: at least one group mean differs

    FIX: Added Tukey HSD post-hoc test to identify WHICH groups differ.
    """
    st.markdown("### One-Way ANOVA")
    st.write("Tests whether 3+ group means are significantly different.")

    numeric_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)

    if not cat_cols:
        st.error("One-Way ANOVA requires a categorical grouping column.")
        return
    if not numeric_cols:
        st.error("No numeric columns available.")
        return

    group_col = st.selectbox("Grouping column:", cat_cols, key="anova1_grp")
    num_col = st.selectbox("Numeric column:", numeric_cols, key="anova1_num")
    alpha = st.selectbox("Significance level (α):", [0.05, 0.01, 0.10], key="anova1_alpha")

    if st.button("Run One-Way ANOVA"):
        group_labels = df[group_col].dropna().unique()
        groups = []
        valid_labels = []

        for g in group_labels:
            grp_data = drop_missing(df[df[group_col] == g][num_col])
            if len(grp_data) >= 2:
                groups.append(grp_data)
                valid_labels.append(g)

        if len(groups) < 2:
            st.error("Need at least 2 groups with 2+ values each.")
            return

        f_stat, p_value = sp.f_oneway(*groups)
        k = len(groups)
        n_total = sum(len(g) for g in groups)
        df_between = k - 1
        df_within = n_total - k

        _display_test_results(
            test_name="One-Way ANOVA",
            statistic_name="F-statistic",
            statistic=f_stat,
            p_value=p_value,
            df=f"{df_between}, {df_within}",
            alpha=alpha,
            h0=f"H₀: All group means of '{num_col}' are equal",
            h1=f"H₁: At least one group mean is different",
            extra_info={
                "Groups compared": k,
                "Total observations (N)": n_total,
                "df_between (k-1)": df_between,
                "df_within (N-k)": df_within,
            }
        )

        # Show group means table
        st.markdown("**Group Means:**")
        means_df = pd.DataFrame({
            "Group": [str(g) for g in valid_labels],
            "Mean": [round(grp.mean(), 4) for grp in groups],
            "Std Dev": [round(grp.std(ddof=1), 4) for grp in groups],
            "n": [len(grp) for grp in groups],
        })
        st.dataframe(means_df, use_container_width=True, hide_index=True)

        # FIX: Tukey HSD post-hoc test — shows WHICH pairs differ
        if p_value < alpha:
            st.markdown("---")
            st.markdown("#### 🔍 Tukey HSD Post-hoc Test")
            st.write("""
            ANOVA only tells you *that* groups differ — not *which* ones.
            Tukey HSD (Honestly Significant Difference) compares every pair of groups.
            """)
            try:
                # Build arrays for Tukey: values and group labels
                all_values = np.concatenate(groups)
                all_labels = np.concatenate([[str(lbl)] * len(grp)
                                              for lbl, grp in zip(valid_labels, groups)])

                tukey = pairwise_tukeyhsd(all_values, all_labels, alpha=alpha)

                # Convert Tukey results to DataFrame
                tukey_df = pd.DataFrame(
                    data=tukey._results_table.data[1:],
                    columns=tukey._results_table.data[0]
                )
                # Rename columns for clarity
                tukey_df.columns = ["Group 1", "Group 2", "Mean Diff", "p-adj", "Lower CI", "Upper CI", "Reject H0"]

                # Round numerics
                for c in ["Mean Diff", "p-adj", "Lower CI", "Upper CI"]:
                    tukey_df[c] = tukey_df[c].apply(lambda x: round(float(x), 4))

                st.dataframe(tukey_df, use_container_width=True, hide_index=True)

                # Plain English summary
                st.markdown("**Summary:**")
                sig_pairs = tukey_df[tukey_df["Reject H0"] == True]
                if len(sig_pairs) > 0:
                    for _, row in sig_pairs.iterrows():
                        st.success(
                            f"✅ **{row['Group 1']}** vs **{row['Group 2']}**: "
                            f"mean difference = {row['Mean Diff']:.2f}, "
                            f"p-adj = {row['p-adj']:.4f} → Significantly different"
                        )
                else:
                    st.info("No specific pair was found to be significantly different (Tukey HSD).")

            except Exception as e:
                st.warning(f"Tukey HSD could not run: {e}")
        else:
            st.info("ℹ️ ANOVA was not significant — post-hoc test not needed.")


def _two_way_anova(df: pd.DataFrame):
    """
    Two-Way ANOVA.
    Tests effect of two categorical factors and their interaction on a numeric outcome.
    """
    st.markdown("### Two-Way ANOVA")
    st.write("Tests the effect of two categorical variables and their interaction on a numeric outcome.")

    numeric_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)

    if len(cat_cols) < 2:
        st.error("Two-Way ANOVA requires at least 2 categorical columns.")
        return
    if not numeric_cols:
        st.error("No numeric columns available.")
        return

    factor1 = st.selectbox("Factor 1 (categorical):", cat_cols, key="anova2_f1")
    factor2 = st.selectbox("Factor 2 (categorical):",
                            [c for c in cat_cols if c != factor1], key="anova2_f2")
    outcome = st.selectbox("Outcome (numeric):", numeric_cols, key="anova2_out")

    if st.button("Run Two-Way ANOVA"):
        try:
            model_df = df[[factor1, factor2, outcome]].dropna()

            if len(model_df) < 10:
                st.error("Need at least 10 complete rows.")
                return

            formula = f"`{outcome}` ~ C(`{factor1}`) + C(`{factor2}`) + C(`{factor1}`):C(`{factor2}`)"
            model = ols(formula, data=model_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            st.markdown("**ANOVA Table (Type II SS):**")
            st.dataframe(anova_table.round(4), use_container_width=True)

            st.markdown("**Interpretation:**")
            for idx in anova_table.index[:-1]:
                p = anova_table.loc[idx, "PR(>F)"]
                F = anova_table.loc[idx, "F"]
                if p < 0.05:
                    st.success(f"✅ `{idx}`: F={F:.4f}, p={format_pvalue(p)} → **Significant effect**")
                else:
                    st.info(f"ℹ️ `{idx}`: F={F:.4f}, p={format_pvalue(p)} → No significant effect")

        except Exception as e:
            st.error(f"Two-Way ANOVA failed: {e}")
            st.warning("Make sure each factor has ≥2 levels and the dataset has enough observations per cell.")


def _chi_square_test(df: pd.DataFrame):
    """
    Chi-Square Test of Independence.
    H0: variables are independent
    H1: variables are associated
    """
    st.markdown("### Chi-Square Test of Independence")
    st.write("Tests whether two categorical variables are associated.")

    cat_cols = get_categorical_columns(df)

    if len(cat_cols) < 2:
        st.error("Chi-Square test requires at least 2 categorical columns.")
        return

    var1 = st.selectbox("Variable 1:", cat_cols, key="chi_v1")
    var2 = st.selectbox("Variable 2:", [c for c in cat_cols if c != var1], key="chi_v2")
    alpha = st.selectbox("Significance level (α):", [0.05, 0.01, 0.10], key="chi_alpha")

    if st.button("Run Chi-Square Test"):
        clean_df = df[[var1, var2]].dropna()

        if len(clean_df) < 5:
            st.error("Need at least 5 complete observations.")
            return

        contingency_table = pd.crosstab(clean_df[var1], clean_df[var2])

        st.markdown("**Contingency Table (Observed Frequencies):**")
        st.dataframe(contingency_table, use_container_width=True)

        chi2, p_value, dof, expected = sp.chi2_contingency(contingency_table)

        low_expected = (expected < 5).sum()
        if low_expected > 0:
            st.warning(
                f"⚠️ {low_expected} cell(s) have expected frequency < 5. "
                "Chi-square assumption violated — results may not be reliable. "
                "Consider merging categories or using Fisher's exact test."
            )

        _display_test_results(
            test_name="Chi-Square Test of Independence",
            statistic_name="χ² statistic",
            statistic=chi2,
            p_value=p_value,
            df=dof,
            alpha=alpha,
            h0=f"H₀: '{var1}' and '{var2}' are independent",
            h1=f"H₁: '{var1}' and '{var2}' are NOT independent (associated)",
            extra_info={
                "dof = (rows-1) × (cols-1)": f"({contingency_table.shape[0]-1}) × ({contingency_table.shape[1]-1}) = {dof}",
                "n (observations)": len(clean_df),
                "Cells with expected < 5": int(low_expected),
            }
        )

        with st.expander("📋 Expected Frequencies"):
            expected_df = pd.DataFrame(
                expected,
                index=contingency_table.index,
                columns=contingency_table.columns
            ).round(2)
            st.dataframe(expected_df, use_container_width=True)


# =====================================================================
# Shared helpers
# =====================================================================

def _display_test_results(test_name, statistic_name, statistic, p_value,
                           df, alpha, h0, h1, extra_info: dict):
    """Display standardized test results with hypotheses, metrics, decision, and extra info."""
    with st.expander(f"📋 Results: {test_name}", expanded=True):
        st.markdown(f"**{h0}**")
        st.markdown(f"**{h1}**")
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(statistic_name, f"{statistic:.4f}")
        with col2:
            st.metric("P-value", format_pvalue(p_value))
        with col3:
            if df is not None:
                st.metric("Degrees of Freedom", str(df))

        st.markdown("---")
        st.markdown(f"**Significance Level (α):** {alpha}")

        if p_value < alpha:
            st.error(
                f"🔴 **Reject H₀** — p-value ({format_pvalue(p_value)}) < α ({alpha})\n\n"
                f"There IS a statistically significant result.\n{h1}"
            )
        else:
            st.success(
                f"🟢 **Fail to Reject H₀** — p-value ({format_pvalue(p_value)}) ≥ α ({alpha})\n\n"
                f"There is NOT enough evidence to reject:\n{h0}"
            )

        if extra_info:
            st.markdown("**Additional Information:**")
            for key, value in extra_info.items():
                st.write(f"- {key}: `{value}`")


def _check_normality(series: pd.Series, col_name: str):
    """Shapiro-Wilk normality check for t-test assumption."""
    with st.expander(f"🔍 Normality Check for '{col_name}'"):
        n = len(series)
        if n > 5000:
            st.info(f"n={n} is large — by CLT the t-test is robust even for non-normal data.")
            return
        stat, p = sp.shapiro(series)
        st.write(f"**Shapiro-Wilk Test:** W = {stat:.4f}, p = {format_pvalue(p)}")
        if p > 0.05:
            st.success("✅ Data appears normally distributed (p > 0.05). T-test assumption met.")
        else:
            st.warning(
                f"⚠️ Data may NOT be normally distributed (p ≤ 0.05). "
                f"T-test is still valid for n={n} ≥ 30 by CLT. "
                f"For small samples, consider Mann-Whitney U test."
            )
