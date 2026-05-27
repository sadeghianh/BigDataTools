# =========================
# app.py
# Main Streamlit application entry point
# This is the file you run: streamlit run app.py
# =========================

# ---- Standard library imports ----
import io                          # For reading file bytes into pandas

# ---- Third-party imports ----
import pandas as pd                # Import pandas for dataset loading and handling
import streamlit as st             # Import streamlit — the web dashboard framework

# ---- Module imports (our own code) ----
from modules.stats import render_statistics          # Descriptive statistics module
from modules.plots import render_plots               # Visualization module
from modules.sampling import render_sampling         # Sampling methods module
from modules.normalization import render_normalization  # Normalization module
from modules.distributions import render_distributions  # Theoretical distributions module
from modules.fitting import render_fitting           # Distribution fitting + CLT module
from modules.tests import render_tests               # Hypothesis testing module
from utils.helpers import validate_dataframe         # DataFrame validation utility


# =====================================================================
# PAGE CONFIGURATION
# Must be the very first Streamlit call in the script
# =====================================================================

st.set_page_config(
    page_title="Statistical Dashboard",   # Browser tab title
    page_icon="📊",                        # Browser tab icon
    layout="wide",                        # Use full screen width
    initial_sidebar_state="expanded",    # Start with sidebar open
)


# =====================================================================
# CUSTOM CSS STYLING
# Injects CSS directly into the Streamlit page for custom look
# =====================================================================

st.markdown("""
<style>
/* ---- Main background color ---- */
.main {
    background-color: #f8f9fb;
}

/* ---- Sidebar styling ---- */
[data-testid="stSidebar"] {
    background-color: #1a1f36;
    color: white;
}

/* ---- Metric card styling ---- */
[data-testid="stMetricValue"] {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1a1f36;
}

/* ---- Section headers ---- */
h3 {
    color: #1a1f36;
}

/* ---- Button styling ---- */
.stButton > button {
    background-color: #3b5bdb;
    color: white;
    border-radius: 6px;
    border: none;
    padding: 0.4rem 1.2rem;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #2f4dc4;
}

/* ---- Expander header ---- */
details > summary {
    font-weight: 600;
    color: #1a1f36;
}
</style>
""", unsafe_allow_html=True)   # unsafe_allow_html=True is required to inject raw HTML/CSS


# =====================================================================
# SIDEBAR — Navigation menu and file upload
# =====================================================================

with st.sidebar:
    # Display the dashboard logo/title in the sidebar
    st.markdown("## 📊 Stats Dashboard")
    st.markdown("*Interactive Statistical Analysis*")
    st.markdown("---")   # Horizontal divider

    # ---- FILE UPLOAD ----
    st.markdown("### 📂 Upload Dataset")

    # st.file_uploader creates a drag-and-drop file upload widget
    uploaded_file = st.file_uploader(
        "Upload a CSV file",         # Label text
        type=["csv"],                # Only accept .csv files
        help="Upload any CSV dataset to begin analysis",  # Tooltip text
    )

    st.markdown("---")   # Divider between upload and navigation

    # ---- NAVIGATION MENU ----
    st.markdown("### 🧭 Navigate")

    # st.radio creates a vertical radio button menu
    page = st.radio(
        "Choose a module:",          # Label
        [                            # List of available pages
            "🏠 Home",
            "📐 Descriptive Statistics",
            "📈 Visualizations",
            "🎲 Sampling",
            "⚖️ Normalization",
            "🔔 Distributions",
            "🔗 Fitting & CLT",
            "🧪 Hypothesis Testing",
        ],
        label_visibility="collapsed",   # Hide the label (already shown as heading above)
        key="nav_radio"
    )

    st.markdown("---")   # Bottom divider

    # Show dataset info in sidebar if a file is loaded
    if uploaded_file is not None:
        # Display a green indicator when file is loaded
        st.markdown("### ✅ Dataset Loaded")
        # These will be filled in after loading below


# =====================================================================
# DATASET LOADING
# Load the uploaded CSV into a pandas DataFrame and store in session
# =====================================================================

# st.session_state is a dictionary that persists across Streamlit reruns
# We use it to store the loaded DataFrame so it doesn't disappear on interaction

if "df" not in st.session_state:
    st.session_state["df"] = None   # Initialize to None if not yet loaded

if uploaded_file is not None:
    try:
        # Read the uploaded file's bytes as a string buffer
        # This is needed because Streamlit's file_uploader returns a BytesIO object
        bytes_data = uploaded_file.read()   # Read raw bytes from the uploaded file

        # Try UTF-8 first, fall back to latin-1 for special characters
        try:
            df_loaded = pd.read_csv(io.BytesIO(bytes_data), encoding="utf-8")   # Parse CSV
        except UnicodeDecodeError:
            df_loaded = pd.read_csv(io.BytesIO(bytes_data), encoding="latin-1")  # Fallback

        # Save the DataFrame into session state so all pages can access it
        st.session_state["df"] = df_loaded

    except Exception as e:
        # If parsing fails, show a friendly error message
        st.sidebar.error(f"Could not read file: {e}")

# Retrieve the DataFrame from session state (may be None if no file loaded)
df = st.session_state.get("df", None)

# Update the sidebar with dataset info if loaded
if df is not None:
    with st.sidebar:
        st.markdown(f"- **Rows:** {df.shape[0]}")             # Show row count
        st.markdown(f"- **Columns:** {df.shape[1]}")          # Show column count
        st.markdown(f"- **Numeric cols:** {len(df.select_dtypes(include='number').columns)}")  # Numeric count
        missing_pct = 100 * df.isnull().sum().sum() / (df.shape[0] * df.shape[1])  # % missing
        st.markdown(f"- **Missing values:** {missing_pct:.1f}%")  # Show missing %


# =====================================================================
# PAGE ROUTING
# Display the correct module based on sidebar navigation selection
# =====================================================================

# ---- HOME PAGE ----
if page == "🏠 Home":
    # Display the main dashboard welcome page
    st.title("📊 Statistical Analysis Dashboard")
    st.markdown("### Welcome to your interactive statistics toolbox")
    st.markdown("""
    This dashboard provides a complete statistical analysis workflow for any CSV dataset.
    Upload a file using the sidebar and navigate through the modules below.
    """)

    # Display a feature overview grid using Streamlit columns
    col1, col2, col3 = st.columns(3)   # Three columns for feature cards

    with col1:
        st.info("**📐 Descriptive Statistics**\nMean, Median, Mode, Variance, Std Dev with formulas")
        st.info("**📈 Visualizations**\nHistogram, Boxplot, Scatter, KDE, Spider, Gauge & more")
        st.info("**🎲 Sampling**\nRandom, Systematic, Stratified sampling methods")

    with col2:
        st.info("**⚖️ Normalization**\nMin-Max scaling and Z-score standardization")
        st.info("**🔔 Distributions**\nNormal, Poisson, Exponential, Binomial, Bernoulli, Uniform")
        st.info("**🔗 Fitting & CLT**\nFit your data to distributions + CLT simulation")

    with col3:
        st.info("**🧪 Hypothesis Testing**\nT-test, Z-test, ANOVA, Chi-square with interpretation")
        st.warning("**📂 Getting Started**\nUpload a CSV file using the sidebar to begin analysis")
        st.success("**✅ All modules are fully interactive and educational**")

    # If no file is uploaded, show a demo prompt
    if df is None:
        st.markdown("---")
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. Click **Browse files** in the sidebar
        2. Upload any CSV dataset (e.g., Iris, Titanic, or your own)
        3. Use the sidebar menu to navigate to any module
        4. Select columns and click buttons to explore your data
        """)
        st.info("💡 Tip: You can download sample datasets from [Kaggle](https://www.kaggle.com/datasets) or use any CSV file.")
    else:
        # Show a preview of the loaded dataset
        st.markdown("---")
        st.markdown("### 📋 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)   # Show first 10 rows
        st.caption(f"Showing first 10 of {len(df)} rows · {len(df.columns)} columns")

        # Show basic data info
        with st.expander("📊 Column Data Types"):
            # Create a summary of column names and their data types
            dtype_df = pd.DataFrame({
                "Column": df.dtypes.index,                   # Column names
                "Data Type": df.dtypes.values.astype(str),  # Their dtypes as strings
                "Non-Null Count": df.count().values,         # Count of non-null values
                "Null Count": df.isnull().sum().values,      # Count of null values
            })
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)

# ---- STATISTICS PAGE ----
elif page == "📐 Descriptive Statistics":
    if validate_dataframe(df):      # Validate that dataset is loaded
        render_statistics(df)       # Call the statistics module

# ---- VISUALIZATIONS PAGE ----
elif page == "📈 Visualizations":
    if validate_dataframe(df):      # Validate dataset
        render_plots(df)            # Call the plots module

# ---- SAMPLING PAGE ----
elif page == "🎲 Sampling":
    if validate_dataframe(df):      # Validate dataset
        render_sampling(df)         # Call the sampling module

# ---- NORMALIZATION PAGE ----
elif page == "⚖️ Normalization":
    if validate_dataframe(df):      # Validate dataset
        render_normalization(df)    # Call the normalization module

# ---- DISTRIBUTIONS PAGE ----
elif page == "🔔 Distributions":
    # Distributions module doesn't need a dataset — it's theoretical
    render_distributions(df)        # Call distributions module (df may be None here)

# ---- FITTING & CLT PAGE ----
elif page == "🔗 Fitting & CLT":
    if validate_dataframe(df):      # Validate dataset (CLT simulation needs data)
        render_fitting(df)          # Call the fitting and CLT module

# ---- HYPOTHESIS TESTING PAGE ----
elif page == "🧪 Hypothesis Testing":
    if validate_dataframe(df):      # Validate dataset
        render_tests(df)            # Call the hypothesis testing module


# =====================================================================
# FOOTER
# =====================================================================

st.markdown("---")   # Bottom divider
st.markdown(
    "<div style='text-align:center; color:#999; font-size:0.8rem;'>"
    "Statistical Dashboard · Built with Python & Streamlit · Educational Use"
    "</div>",
    unsafe_allow_html=True   # Allow HTML for centered/styled footer text
)
