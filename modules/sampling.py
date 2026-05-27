# =========================
# modules/sampling.py
# Sampling module — uses real dataset
# Fix: replaced groupby().apply() with manual loop to avoid pandas FutureWarning
# =========================

import pandas as pd
import numpy as np
import streamlit as st
from utils.helpers import get_categorical_columns, section_header


def render_sampling(df: pd.DataFrame):
    section_header("Sampling Methods", "🎲")

    st.write(f"Dataset has **{len(df)} rows** and **{len(df.columns)} columns**.")

    method = st.selectbox(
        "Select sampling method:",
        ["Random Sampling", "Systematic Sampling", "Stratified Sampling"],
        key="sampling_method"
    )

    if method == "Random Sampling":
        n = st.slider("Number of samples:", 1, len(df), min(50, len(df)), key="rand_n")
        if st.button("Draw Random Sample"):
            _random_sampling(df, n)

    elif method == "Systematic Sampling":
        k = st.slider("Select every k-th row (step size):", 2, max(2, len(df)//2), 5, key="sys_k")
        if st.button("Draw Systematic Sample"):
            _systematic_sampling(df, k)

    elif method == "Stratified Sampling":
        cat_cols = get_categorical_columns(df)
        if not cat_cols:
            st.warning("Stratified sampling requires at least one categorical column.")
        else:
            strat_col = st.selectbox("Select stratification column:", cat_cols, key="strat_col")
            frac = st.slider("Fraction per stratum:", 0.05, 1.0, 0.3, 0.05, key="strat_frac")
            if st.button("Draw Stratified Sample"):
                _stratified_sampling(df, strat_col, frac)


def _random_sampling(df: pd.DataFrame, n: int):
    """Simple random sampling — each row equally likely to be selected."""
    sample = df.sample(n=n, random_state=42)

    with st.expander(f"✅ Random Sample — {n} rows", expanded=True):
        st.dataframe(sample, use_container_width=True)
        st.caption(f"Sampled {n} rows from {len(df)} total ({100*n/len(df):.1f}%)")
        st.info(f"""
        **Method:** Simple Random Sampling
        **Formula:** P(row i selected) = n / N = {n} / {len(df)} = {n/len(df):.4f}
        **Result:** {n} rows drawn randomly without replacement (random_state=42 for reproducibility)
        """)
        csv = sample.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Sample CSV", csv, "random_sample.csv", "text/csv")


def _systematic_sampling(df: pd.DataFrame, k: int):
    """Systematic sampling — every k-th row from a random start."""
    np.random.seed(42)
    start = np.random.randint(0, k)
    indices = np.arange(start, len(df), k)
    sample = df.iloc[indices]

    with st.expander(f"✅ Systematic Sample — every {k}th row", expanded=True):
        st.dataframe(sample, use_container_width=True)
        st.caption(f"Start={start}, step={k}. Total: {len(sample)} rows.")
        st.info(f"""
        **Method:** Systematic Sampling
        **Formula:** Selected rows = start, start+k, start+2k, ...
        **Parameters used:** start={start} (random), k={k}
        **Result:** {len(sample)} rows selected from {len(df)} total
        """)
        csv = sample.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Sample CSV", csv, "systematic_sample.csv", "text/csv")


def _stratified_sampling(df: pd.DataFrame, strat_col: str, frac: float):
    """
    Stratified sampling — sample proportionally from each group.
    FIX: Uses manual loop instead of groupby().apply() to avoid pandas FutureWarning.
    """
    try:
        # Manual loop per group — avoids pandas groupby().apply() FutureWarning
        chunks = []
        group_info = {}

        for group_name, group_df in df.groupby(strat_col):
            n_group = len(group_df)
            n_sample = max(1, int(np.floor(n_group * min(1.0, frac))))
            sampled = group_df.sample(n=n_sample, random_state=42)
            chunks.append(sampled)
            group_info[str(group_name)] = {
                "Total in group": n_group,
                "Sampled": n_sample,
                "Fraction": f"{100*n_sample/n_group:.1f}%"
            }

        sample = pd.concat(chunks).reset_index(drop=True)

        with st.expander(f"✅ Stratified Sample by '{strat_col}'", expanded=True):
            st.dataframe(sample, use_container_width=True)

            st.markdown("**Sampling details per stratum:**")
            info_df = pd.DataFrame(group_info).T.reset_index()
            info_df.columns = ["Group", "Total in group", "Sampled", "Fraction"]
            st.dataframe(info_df, use_container_width=True, hide_index=True)

            st.caption(f"Total sampled: {len(sample)} rows from {len(df)} total ({100*len(sample)/len(df):.1f}%)")
            st.info(f"""
            **Method:** Stratified Sampling
            **Formula:** n_i = floor(N_i × frac) for each stratum i
            **Fraction used:** {int(frac*100)}% per group
            **Result:** {len(sample)} rows total, proportionally from each '{strat_col}' group
            """)
            csv = sample.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Sample CSV", csv, "stratified_sample.csv", "text/csv")

    except Exception as e:
        st.error(f"Stratified sampling failed: {e}")
