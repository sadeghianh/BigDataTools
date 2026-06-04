# =========================
# app.py
# Main Streamlit application entry point
# This is the file you run:  streamlit run app.py
# =========================

# ---- Standard library imports ----
import io                          # For reading uploaded file bytes into pandas

# ---- Third-party imports ----
import pandas as pd                # Import pandas for dataset loading and handling
import streamlit as st             # Import streamlit — the web dashboard framework

# ---- Module imports (our own code) ----
from modules.stats import render_statistics             # Descriptive statistics module
from modules.plots import render_plots                  # Visualization module
from modules.sampling import render_sampling            # Sampling methods module
from modules.normalization import render_normalization  # Normalization module
from modules.distributions import render_distributions  # Theoretical distributions module
from modules.fitting import render_fitting              # Distribution fitting + CLT module
from modules.tests import render_tests                  # Hypothesis testing module
from modules.data_profile import render_data_profile    # Data profile ("know your data") module
from utils.helpers import (                             # Shared helper functions
    validate_dataframe,                                 # Checks a DataFrame is loaded and non-empty
    render_unit_manager,                                # Sidebar unit inputs for numeric columns
    srh_logo_svg,                                       # Inline SRH University logo (SVG)
)


# =====================================================================
# PAGE CONFIGURATION
# Must be the very first Streamlit call in the script
# =====================================================================

st.set_page_config(                       # Configure the Streamlit page settings
    page_title="Statistical Dashboard",   # Browser tab title
    page_icon="📊",                        # Browser tab icon
    layout="wide",                         # Use the full screen width
    initial_sidebar_state="expanded",      # Start with the sidebar open
)


# =====================================================================
# CUSTOM CSS STYLING — light, modern, SRH-branded theme
# Injects CSS directly into the page for a polished look
# =====================================================================

st.markdown("""
<style>
/* ---- Import a clean Google font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ---- Global font ---- */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

/* ---- Light gradient main background ---- */
.stApp {
    background: linear-gradient(135deg, #f6f9fc 0%, #eef3f9 100%);
}

/* ---- Sidebar: soft white with subtle border ---- */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e6ebf2;
}
[data-testid="stSidebar"] * {
    color: #2c3e50;
}

/* ---- Sidebar navigation radio buttons as clean cards ---- */
[data-testid="stSidebar"] .stRadio > div {
    gap: 4px;
}
[data-testid="stSidebar"] .stRadio label {
    background: #f7fafc;
    border: 1px solid #e6ebf2;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 2px;
    transition: all 0.15s ease;
    font-weight: 500;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: #fff0e9;
    border-color: #D44407;
}

/* ---- Metric cards: white with soft shadow and orange accent ---- */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e6ebf2;
    border-left: 4px solid #D44407;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(16,42,67,0.06);
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem;
    font-weight: 700;
    color: #102a43;
}
[data-testid="stMetricLabel"] {
    font-weight: 500;
    color: #627d98;
}

/* ---- Headings ---- */
h1, h2, h3 {
    color: #102a43;
    font-weight: 700;
}

/* ---- Primary buttons: SRH orange ---- */
.stButton > button {
    background: linear-gradient(135deg, #D44407 0%, #e85d20 100%);
    color: #ffffff;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1.4rem;
    font-weight: 600;
    box-shadow: 0 2px 6px rgba(212,68,7,0.25);
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #b8390a 0%, #D44407 100%);
    box-shadow: 0 4px 10px rgba(212,68,7,0.35);
    transform: translateY(-1px);
}

/* ---- Tabs / radio inside main area ---- */
.stRadio [role="radiogroup"] label {
    font-weight: 500;
}

/* ---- Expander: card-like ---- */
details {
    background: #ffffff;
    border: 1px solid #e6ebf2;
    border-radius: 10px;
    padding: 4px 12px;
    box-shadow: 0 1px 3px rgba(16,42,67,0.05);
}
details > summary {
    font-weight: 600;
    color: #102a43;
}

/* ---- DataFrame container ---- */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e6ebf2;
}

/* ---- Info/success/warning boxes: rounder ---- */
.stAlert {
    border-radius: 10px;
}

/* ---- Selectbox & inputs ---- */
.stSelectbox > div > div, .stTextInput > div > div {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)   # unsafe_allow_html=True is required to inject raw CSS


# =====================================================================
# SIDEBAR — Logo, file upload, and navigation menu
# =====================================================================

with st.sidebar:  # Everything inside here renders in the sidebar
    # ---- SRH University logo (inline SVG, official orange) ----
    st.markdown(srh_logo_svg(width=170), unsafe_allow_html=True)  # Render the SRH logo

    # ---- Dashboard title ----
    st.markdown("## 📊 Stats Dashboard")            # Dashboard title
    st.caption("Interactive Statistical Analysis · SRH University")  # Subtitle
    st.markdown("---")  # Divider

    # ---- FILE UPLOAD ----
    st.markdown("### 📂 Upload Dataset")  # Upload section heading
    uploaded_file = st.file_uploader(     # Drag-and-drop CSV upload widget
        "Upload a CSV file",              # Label
        type=["csv"],                     # Only accept .csv files
        help="Upload any CSV dataset to begin analysis",  # Tooltip
    )

    st.markdown("---")  # Divider between upload and navigation

    # ---- NAVIGATION MENU ----
    st.markdown("### 🧭 Navigate")  # Navigation heading
    page = st.radio(                # Vertical radio-button menu
        "Choose a module:",         # Label (hidden below)
        [                           # List of available pages
            "🏠 Home",                      # Welcome page
            "🔎 Data Profile",              # Know-your-data overview (NEW)
            "📐 Descriptive Statistics",    # Mean, median, std, etc.
            "📈 Visualizations",            # Charts
            "🎲 Sampling",                  # Sampling methods
            "⚖️ Normalization",             # Scaling methods
            "🔔 Distributions",             # Theoretical distributions
            "🔗 Fitting & CLT",             # Distribution fitting + CLT
            "🧪 Hypothesis Testing",        # All statistical tests
        ],
        label_visibility="collapsed",   # Hide the label (heading shown above)
        key="nav_radio"                 # Unique widget key
    )


# =====================================================================
# DATASET LOADING
# Load the uploaded CSV into a pandas DataFrame and keep it in session state
# =====================================================================

if "df" not in st.session_state:    # First run — no DataFrame yet
    st.session_state["df"] = None   # Initialize to None

if uploaded_file is not None:       # A file has been uploaded
    try:
        bytes_data = uploaded_file.read()  # Read the raw bytes from the upload
        # Try UTF-8 first; fall back to latin-1 for files with special characters
        try:
            df_loaded = pd.read_csv(io.BytesIO(bytes_data), encoding="utf-8")  # Parse as UTF-8
        except UnicodeDecodeError:  # UTF-8 failed
            df_loaded = pd.read_csv(io.BytesIO(bytes_data), encoding="latin-1")  # Fallback encoding
        st.session_state["df"] = df_loaded  # Store the loaded DataFrame in session state
    except Exception as e:  # Any parsing error
        st.sidebar.error(f"Could not read file: {e}")  # Show a friendly error in the sidebar

df = st.session_state.get("df", None)  # Retrieve the DataFrame (None if nothing loaded)

# ---- Sidebar dataset info + unit manager (only when a file is loaded) ----
if df is not None:  # A dataset is loaded
    with st.sidebar:  # Render in the sidebar
        st.markdown("---")  # Divider
        st.markdown("### ✅ Dataset Loaded")  # Status heading
        st.markdown(f"- **Rows:** {df.shape[0]:,}")          # Row count
        st.markdown(f"- **Columns:** {df.shape[1]}")         # Column count
        st.markdown(f"- **Numeric cols:** {len(df.select_dtypes(include='number').columns)}")  # Numeric count
        missing_pct = 100 * df.isnull().sum().sum() / (df.shape[0] * df.shape[1])  # % missing cells
        st.markdown(f"- **Missing values:** {missing_pct:.1f}%")  # Missing percentage

    # Render the unit manager so the user can define units for each numeric column.
    # These units appear on chart axes and in results throughout the dashboard.
    render_unit_manager(df)  # Show unit text inputs in the sidebar


# =====================================================================
# PAGE ROUTING — show the selected module
# =====================================================================

# ---- HOME PAGE ----
if page == "🏠 Home":  # Home selected
    st.title("📊 Statistical Analysis Dashboard")  # Main page title
    st.markdown("#### Welcome to your interactive statistics toolbox — built for SRH University")  # Subtitle

    st.markdown("This dashboard provides a complete statistical analysis workflow for any CSV dataset. "
                "Upload a file using the sidebar and explore the modules below.")  # Intro text

    # Feature overview cards in three columns
    col1, col2, col3 = st.columns(3)  # Three columns

    with col1:  # First column of feature cards
        st.info("**🔎 Data Profile**\nUnderstand your data: size, types, missing values, normality")  # Card
        st.info("**📐 Descriptive Statistics**\nMean, Median, Mode, Variance, Std Dev with formulas")  # Card
        st.info("**📈 Visualizations**\nHistogram, Boxplot, Scatter, KDE, Violin & more")  # Card

    with col2:  # Second column
        st.info("**🎲 Sampling**\nRandom, Systematic, Stratified sampling methods")  # Card
        st.info("**⚖️ Normalization**\nMin-Max scaling and Z-score standardization")  # Card
        st.info("**🔔 Distributions**\nNormal, Poisson, Exponential, Binomial, Bernoulli, Uniform")  # Card

    with col3:  # Third column
        st.info("**🔗 Fitting & CLT**\nFit your data to distributions + CLT simulation")  # Card
        st.info("**🧪 Hypothesis Testing**\nParametric & non-parametric tests + a Test Advisor")  # Card
        st.success("**✅ Every module is interactive, uses your real data, and explains its results**")  # Card

    # If no dataset is loaded, show quick-start instructions; otherwise show a preview
    if df is None:  # No data loaded yet
        st.markdown("---")  # Divider
        st.markdown("### 🚀 Quick Start")  # Quick start heading
        st.markdown("1. Click **Browse files** in the sidebar\n"
                    "2. Upload any CSV dataset (e.g. Iris, Titanic, or your own)\n"
                    "3. Visit **Data Profile** to understand your data\n"
                    "4. Use the menu to run any analysis")  # Numbered steps
        st.info("💡 Tip: You can download sample datasets from Kaggle or use any CSV file.")  # Tip
    else:  # A dataset is loaded
        st.markdown("---")  # Divider
        st.markdown("### 📋 Dataset Preview")  # Preview heading
        st.dataframe(df.head(10), use_container_width=True)  # Show first 10 rows
        st.caption(f"Showing first 10 of {len(df):,} rows · {len(df.columns)} columns")  # Caption

        with st.expander("📊 Column Data Types"):  # Collapsible column-type table
            dtype_df = pd.DataFrame({  # Build a summary of columns and dtypes
                "Column": df.dtypes.index,                    # Column names
                "Data Type": df.dtypes.values.astype(str),    # Data types as strings
                "Non-Null Count": df.count().values,          # Non-null counts
                "Null Count": df.isnull().sum().values,       # Null counts
            })  # End DataFrame
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)  # Show the table

# ---- DATA PROFILE PAGE ----
elif page == "🔎 Data Profile":  # Data Profile selected
    if validate_dataframe(df):     # Ensure a dataset is loaded
        render_data_profile(df)    # Call the data profile module

# ---- DESCRIPTIVE STATISTICS PAGE ----
elif page == "📐 Descriptive Statistics":  # Statistics selected
    if validate_dataframe(df):              # Ensure dataset is loaded
        render_statistics(df)               # Call the statistics module

# ---- VISUALIZATIONS PAGE ----
elif page == "📈 Visualizations":  # Visualizations selected
    if validate_dataframe(df):      # Ensure dataset is loaded
        render_plots(df)            # Call the plots module

# ---- SAMPLING PAGE ----
elif page == "🎲 Sampling":  # Sampling selected
    if validate_dataframe(df):  # Ensure dataset is loaded
        render_sampling(df)     # Call the sampling module

# ---- NORMALIZATION PAGE ----
elif page == "⚖️ Normalization":  # Normalization selected
    if validate_dataframe(df):     # Ensure dataset is loaded
        render_normalization(df)   # Call the normalization module

# ---- DISTRIBUTIONS PAGE ----
elif page == "🔔 Distributions":  # Distributions selected
    render_distributions(df)       # Call distributions module (works with or without data)

# ---- FITTING & CLT PAGE ----
elif page == "🔗 Fitting & CLT":  # Fitting & CLT selected
    if validate_dataframe(df):      # Ensure dataset is loaded (CLT needs data)
        render_fitting(df)          # Call the fitting + CLT module

# ---- HYPOTHESIS TESTING PAGE ----
elif page == "🧪 Hypothesis Testing":  # Hypothesis testing selected
    if validate_dataframe(df):          # Ensure dataset is loaded
        render_tests(df)                # Call the hypothesis testing module


# =====================================================================
# FOOTER
# =====================================================================

st.markdown("---")  # Bottom divider
st.markdown(  # Centered footer text
    "<div style='text-align:center; color:#9fb3c8; font-size:0.8rem; padding:8px;'>"  # Styled container
    "Statistical Dashboard · SRH University · Built with Python &amp; Streamlit · Educational Use"  # Footer text
    "</div>",
    unsafe_allow_html=True   # Allow HTML for the centered footer
)
