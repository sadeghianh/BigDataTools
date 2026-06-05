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
from modules.confidence import render_confidence_intervals  # Confidence intervals module
from modules.data_profile import render_data_profile    # Data profile ("know your data") module
from utils.helpers import (                             # Shared helper functions
    validate_dataframe,                                 # Checks a DataFrame is loaded and non-empty
    render_unit_manager,                                # Unit inputs for numeric columns
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
# CUSTOM CSS STYLING — modern "Slate & Amber" theme
# A deep slate sidebar with warm amber accents and airy light content
# =====================================================================

st.markdown("""
<style>
/* ---- Clean modern font ---- */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ====== MAIN CONTENT BACKGROUND ====== */
.stApp {
    background:
      radial-gradient(1200px 600px at 80% -10%, #fff4ec 0%, rgba(255,244,236,0) 55%),
      radial-gradient(900px 500px at -10% 10%, #eaf6f4 0%, rgba(234,246,244,0) 50%),
      #f4f7fa;
}

/* Constrain main content width so it doesn't stretch on wide screens */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1280px;
}

/* ====== SIDEBAR — deep slate gradient ====== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2a3f 0%, #143b54 100%);
    border-right: none;
}
/* Default sidebar text = light */
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #dbe7f0 !important;
}
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #8fb0c4 !important;
}

/* ====== FILE UPLOADER — readable on the dark sidebar ====== */
/* The uploader's drop zone: give it a solid light card look */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #ffffff;
    border: 2px dashed #c2d3e0;
    border-radius: 12px;
}
/* All text inside the drop zone = dark slate (so "Drag and drop", size limit, etc. are readable) */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] * {
    color: #1f3b52 !important;
}
/* The little instruction sub-text (e.g. "200MB per file") slightly muted but still readable */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #5b7a91 !important;
}
/* The "Browse files" button inside the uploader = amber, white text */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #f7934c 0%, #e8650e 100%) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 700 !important;
}
/* The uploader's outer label ("Upload a CSV file") stays light on the dark sidebar */
[data-testid="stSidebar"] [data-testid="stFileUploader"] > label,
[data-testid="stSidebar"] [data-testid="stFileUploader"] > label * {
    color: #dbe7f0 !important;
}
/* Uploaded-file chip (after a file is selected): readable dark text on light chip */
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] * {
    color: #1f3b52 !important;
}

/* ====== SIDEBAR NAV: radio rendered as pill buttons ====== */
[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 6px;
    display: flex;
    flex-direction: column;
}
/* hide the little radio circle */
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
    display: none;
}
[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 0;
    cursor: pointer;
    transition: all 0.15s ease;
    width: 100%;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(247,147,76,0.16);
    border-color: rgba(247,147,76,0.5);
}
/* The SELECTED nav item — amber fill, white text (fixes unreadable text) */
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, #f7934c 0%, #e8650e 100%);
    border-color: #e8650e;
    box-shadow: 0 4px 14px rgba(232,101,14,0.45);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* ====== MAIN-AREA RADIO (e.g. test category) — readable chips ====== */
section.main [role="radiogroup"] {
    gap: 8px;
}
section.main [role="radiogroup"] label {
    background: #ffffff;
    border: 1.5px solid #dde6ee;
    border-radius: 10px;
    padding: 8px 16px;
    transition: all 0.15s ease;
    color: #21425c !important;
    font-weight: 600;
}
section.main [role="radiogroup"] label:hover {
    border-color: #f7934c;
    background: #fff8f2;
}
section.main [role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, #16607a 0%, #0f4a60 100%);
    border-color: #0f4a60;
    box-shadow: 0 4px 12px rgba(15,74,96,0.3);
}
section.main [role="radiogroup"] label:has(input:checked) * {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* ====== METRIC CARDS ====== */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e7eef5;
    border-radius: 14px;
    padding: 16px 20px;
    box-shadow: 0 4px 16px rgba(16,42,67,0.06);
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, #f7934c, #e8650e);
}
[data-testid="stMetricValue"] {
    font-size: 1.7rem;
    font-weight: 800;
    color: #0f2a3f;
}
[data-testid="stMetricLabel"] {
    font-weight: 600;
    color: #5b7a91;
}

/* ====== HEADINGS ====== */
section.main h1 {
    color: #0f2a3f;
    font-weight: 800;
    letter-spacing: -0.5px;
}
section.main h2, section.main h3 {
    color: #143b54;
    font-weight: 700;
}

/* ====== BUTTONS — amber gradient ====== */
.stButton > button {
    background: linear-gradient(135deg, #f7934c 0%, #e8650e 100%);
    color: #ffffff;
    border-radius: 10px;
    border: none;
    padding: 0.55rem 1.5rem;
    font-weight: 700;
    box-shadow: 0 4px 14px rgba(232,101,14,0.3);
    transition: all 0.18s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #e8650e 0%, #c9530a 100%);
    box-shadow: 0 6px 20px rgba(232,101,14,0.42);
    transform: translateY(-2px);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ====== DOWNLOAD BUTTON — teal variant ====== */
.stDownloadButton > button {
    background: linear-gradient(135deg, #16607a 0%, #0f4a60 100%);
    color: #ffffff;
    border-radius: 10px;
    border: none;
    font-weight: 700;
    box-shadow: 0 4px 14px rgba(15,74,96,0.3);
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #0f4a60 0%, #0a3548 100%);
}

/* ====== EXPANDERS — clean cards ====== */
details {
    background: #ffffff;
    border: 1px solid #e7eef5;
    border-radius: 14px;
    padding: 6px 16px;
    box-shadow: 0 2px 10px rgba(16,42,67,0.05);
    margin-bottom: 8px;
}
details > summary {
    font-weight: 700;
    color: #143b54;
    padding: 6px 0;
}

/* ====== DATAFRAME ====== */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e7eef5;
    box-shadow: 0 2px 10px rgba(16,42,67,0.04);
}

/* ====== ALERT BOXES ====== */
.stAlert {
    border-radius: 12px;
    border: none;
    box-shadow: 0 2px 10px rgba(16,42,67,0.05);
}

/* ====== INPUTS ====== */
.stSelectbox > div > div, .stTextInput > div > div, .stNumberInput > div > div {
    border-radius: 10px;
}

/* ====== TABS ====== */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    font-weight: 600;
    padding: 8px 16px;
}

/* Hide Streamlit default menu/footer for a cleaner look */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)   # unsafe_allow_html=True is required to inject raw CSS


# =====================================================================
# SIDEBAR — Logo, file upload, and navigation menu
# =====================================================================

with st.sidebar:  # Everything inside here renders in the sidebar
    # ---- SRH University logo (inline SVG, official orange) ----
    st.markdown(srh_logo_svg(width=180), unsafe_allow_html=True)  # Render the SRH logo

    st.markdown("### 📊 Stats Dashboard")  # Dashboard title
    st.caption("Interactive Statistical Analysis")  # Subtitle
    st.markdown("")  # Small spacer

    # ---- FILE UPLOAD ----
    uploaded_file = st.file_uploader(     # Drag-and-drop CSV upload widget
        "📂 Upload a CSV file",           # Label
        type=["csv"],                     # Only accept .csv files
        help="Upload any CSV dataset to begin analysis",  # Tooltip
    )

    st.markdown("")  # Spacer

    # ---- NAVIGATION MENU ----
    st.markdown("##### NAVIGATION")  # Navigation heading (small caps style)
    page = st.radio(                # Vertical radio-button menu (styled as pills)
        "Choose a module:",         # Label (hidden below)
        [                           # List of available pages
            "🏠 Home",                      # Welcome page
            "🔎 Data Profile",              # Know-your-data overview
            "📐 Descriptive Statistics",    # Mean, median, std, etc.
            "📈 Visualizations",            # Charts
            "🎲 Sampling",                  # Sampling methods
            "⚖️ Normalization",             # Scaling methods
            "🔔 Distributions",             # Theoretical distributions
            "🔗 Fitting & CLT",             # Distribution fitting + CLT
            "🧪 Hypothesis Testing",        # All statistical tests
            "🎯 Confidence Intervals",      # Confidence intervals for the mean
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

# ---- Sidebar dataset status (compact, no scrolling needed) ----
if df is not None:  # A dataset is loaded
    with st.sidebar:  # Render in the sidebar
        st.markdown("")  # Spacer
        missing_pct = 100 * df.isnull().sum().sum() / (df.shape[0] * df.shape[1])  # % missing cells
        n_numeric = len(df.select_dtypes(include='number').columns)  # Numeric column count
        # Compact one-line status card using HTML
        st.markdown(
            f"""
            <div style="background:rgba(247,147,76,0.12); border:1px solid rgba(247,147,76,0.3);
                        border-radius:10px; padding:10px 14px; margin-top:8px;">
                <div style="color:#f7934c; font-weight:700; font-size:0.85rem; margin-bottom:4px;">
                    ✅ DATASET LOADED
                </div>
                <div style="color:#dbe7f0; font-size:0.8rem; line-height:1.6;">
                    {df.shape[0]:,} rows · {df.shape[1]} cols<br>
                    {n_numeric} numeric · {missing_pct:.1f}% missing
                </div>
            </div>
            """,
            unsafe_allow_html=True  # Allow the HTML status card
        )


# =====================================================================
# PAGE ROUTING — show the selected module
# =====================================================================

# ---- HOME PAGE ----
if page == "🏠 Home":  # Home selected
    # Hero banner using HTML for a polished landing
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0f2a3f 0%,#16607a 100%);
                    border-radius:18px; padding:36px 40px; margin-bottom:8px;
                    box-shadow:0 10px 30px rgba(15,42,63,0.25);">
            <div style="color:#f7934c; font-weight:700; font-size:0.9rem; letter-spacing:2px;">
                SRH UNIVERSITY · TOOLS &amp; METHODS OF DATA ANALYSIS
            </div>
            <div style="color:#ffffff; font-weight:800; font-size:2.1rem; margin-top:8px; letter-spacing:-0.5px;">
                Statistical Analysis Dashboard
            </div>
            <div style="color:#bcd4e2; font-size:1.05rem; margin-top:10px; max-width:680px;">
                A complete, interactive workflow for exploring any dataset — descriptive statistics,
                visualizations, distributions, and a full suite of parametric &amp; non-parametric tests.
            </div>
        </div>
        """,
        unsafe_allow_html=True  # Allow the hero banner HTML
    )

    # If a dataset is loaded, show preview + units up top; else show quick-start
    if df is not None:  # Data is loaded
        # ---- Units panel right on the home page (no sidebar scrolling) ----
        with st.expander("📏 Set Column Units (optional — shown on all chart axes)", expanded=False):  # Collapsible units
            render_unit_manager(df)  # Render the unit inputs in a grid here

        # ---- Dataset preview ----
        st.markdown("#### 📋 Dataset Preview")  # Preview heading
        st.dataframe(df.head(10), use_container_width=True)  # First 10 rows
        st.caption(f"Showing first 10 of {len(df):,} rows · {len(df.columns)} columns")  # Caption

        # ---- Quick stats row ----
        c1, c2, c3, c4 = st.columns(4)  # Four metric columns
        with c1: st.metric("Rows", f"{df.shape[0]:,}")  # Row count
        with c2: st.metric("Columns", str(df.shape[1]))  # Column count
        with c3: st.metric("Numeric", str(len(df.select_dtypes(include='number').columns)))  # Numeric count
        with c4: st.metric("Categorical", str(len(df.select_dtypes(include=['object','category']).columns)))  # Categorical count

        st.info("👉 Use the sidebar to navigate. Start with **🔎 Data Profile** to understand your data, "
                "then run any analysis. The **🧭 Test Advisor** (in Hypothesis Testing) recommends the right test.")  # Guidance
    else:  # No data loaded
        # Feature cards in three columns
        col1, col2, col3 = st.columns(3)  # Three columns
        with col1:  # First
            st.info("**🔎 Data Profile**\n\nSize, types, missing values, and normality of every column")  # Card
            st.info("**📐 Descriptive Statistics**\n\nMean, Median, Mode, Variance, Std Dev with formulas")  # Card
            st.info("**📈 Visualizations**\n\nHistogram, Boxplot, Scatter, KDE, Violin & more")  # Card
        with col2:  # Second
            st.info("**🎲 Sampling**\n\nRandom, Systematic, and Stratified sampling")  # Card
            st.info("**⚖️ Normalization**\n\nMin-Max scaling and Z-score standardization")  # Card
            st.info("**🔔 Distributions**\n\nNormal, Poisson, Exponential, Binomial, Bernoulli, Uniform")  # Card
        with col3:  # Third
            st.info("**🔗 Fitting & CLT**\n\nFit data to distributions + CLT simulation")  # Card
            st.info("**🧪 Hypothesis Testing**\n\nParametric & non-parametric tests + Test Advisor")  # Card
            st.info("**🎯 Confidence Intervals**\n\nEstimate the range that contains the true mean")  # Card
            st.success("**✅ Real data · Clear explanations · Every result interpreted**")  # Card

        st.markdown("#### 🚀 Quick Start")  # Quick-start heading
        st.markdown("1. Click **Browse files** in the sidebar\n"
                    "2. Upload any CSV dataset (e.g. Iris, Titanic, or your own)\n"
                    "3. Visit **🔎 Data Profile** to understand your data\n"
                    "4. Use the menu to run any analysis")  # Steps

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

# ---- CONFIDENCE INTERVALS PAGE ----
elif page == "🎯 Confidence Intervals":  # Confidence intervals selected
    if validate_dataframe(df):            # Ensure dataset is loaded
        render_confidence_intervals(df)   # Call the confidence intervals module


# =====================================================================
# FOOTER
# =====================================================================

st.markdown(
    "<div style='text-align:center; color:#9fb3c8; font-size:0.8rem; padding:18px;'>"  # Styled footer
    "Statistical Dashboard · SRH University · Built with Python &amp; Streamlit"  # Footer text
    "</div>",
    unsafe_allow_html=True   # Allow HTML for the centered footer
)
