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
from utils.helpers import validate_dataframe, render_unit_manager  # DataFrame validation and unit manager


# =====================================================================
# PAGE CONFIGURATION
# Must be the very first Streamlit call in the script
# =====================================================================

# Configure the Streamlit page title, icon, and layout
st.set_page_config(  # Configure Streamlit page settings
    page_title="Statistical Dashboard",   # Browser tab title
    page_icon="📊",                        # Browser tab icon
    layout="wide",                        # Use full screen width
    initial_sidebar_state="expanded",    # Start with sidebar open
)


# =====================================================================
# CUSTOM CSS STYLING
# Injects CSS directly into the Streamlit page for custom look
# =====================================================================

# Render formatted markdown text in the Streamlit UI
st.markdown("""  # Render formatted markdown text in the Streamlit UI
# Execute this operation
<style>  # Execute this statement
# Execute this operation
/* ---- Main background color ---- */  # Execute this statement
# Execute this operation
.main {  # Execute this statement
    background-color: #f8f9fb;
}

# Execute this operation
/* ---- Sidebar styling ---- */  # Execute this statement
# Compute and store the result in [data-testid
[data-testid="stSidebar"] {  # Store result in 
    background-color: #1a1f36;
    # Execute this operation
    color: white;  # Execute this statement
}

# Execute this operation
/* ---- Metric card styling ---- */  # Execute this statement
# Compute and store the result in [data-testid
[data-testid="stMetricValue"] {  # Store result in 
    # Execute this operation
    font-size: 1.6rem;  # Execute this statement
    # Execute this operation
    font-weight: 700;  # Execute this statement
    color: #1a1f36;
}

# Execute this operation
/* ---- Section headers ---- */  # Execute this statement
# Execute this operation
h3 {  # Execute this statement
    color: #1a1f36;
}

# Execute this operation
/* ---- Button styling ---- */  # Execute this statement
# Execute this operation
.stButton > button {  # Execute this statement
    background-color: #3b5bdb;
    # Execute this operation
    color: white;  # Execute this statement
    # Execute this operation
    border-radius: 6px;  # Execute this statement
    # Execute this operation
    border: none;  # Execute this statement
    # Execute this operation
    padding: 0.4rem 1.2rem;  # Execute this statement
    # Execute this operation
    font-weight: 600;  # Execute this statement
}
# Execute this operation
.stButton > button:hover {  # Execute this statement
    background-color: #2f4dc4;
}

# Execute this operation
/* ---- Expander header ---- */  # Execute this statement
# Execute this operation
details > summary {  # Execute this statement
    # Execute this operation
    font-weight: 600;  # Execute this statement
    color: #1a1f36;
}
# Execute this operation
</style>  # Execute this statement
""", unsafe_allow_html=True)   # unsafe_allow_html=True is required to inject raw HTML/CSS


# =====================================================================
# SIDEBAR — Navigation menu and file upload
# =====================================================================

# Open a context manager (auto-closes when done)
with st.sidebar:  # Open context manager
    # Display the dashboard logo/title in the sidebar
    # Render formatted markdown text in the Streamlit UI
    st.markdown("## 📊 Stats Dashboard")  # Render formatted markdown text in the Streamlit UI
    # Render formatted markdown text in the Streamlit UI
    st.markdown("*Interactive Statistical Analysis*")  # Render formatted markdown text in the Streamlit UI
    st.markdown("---")   # Horizontal divider

    # ---- FILE UPLOAD ----
    # Render formatted markdown text in the Streamlit UI
    st.markdown("### 📂 Upload Dataset")  # Render formatted markdown text in the Streamlit UI

    # st.file_uploader creates a drag-and-drop file upload widget
    # Compute and store the result in uploaded_file
    uploaded_file = st.file_uploader(  # Store result in uploaded_file
        "Upload a CSV file",         # Label text
        type=["csv"],                # Only accept .csv files
        help="Upload any CSV dataset to begin analysis",  # Tooltip text
    )

    st.markdown("---")   # Divider between upload and navigation

    # ---- NAVIGATION MENU ----
    # Render formatted markdown text in the Streamlit UI
    st.markdown("### 🧭 Navigate")  # Render formatted markdown text in the Streamlit UI

    # st.radio creates a vertical radio button menu
    # Compute and store the result in page
    page = st.radio(  # Store result in page
        "Choose a module:",          # Label
        [                            # List of available pages
            # Execute this operation
            "🏠 Home",  # Execute this statement
            # Execute this operation
            "📐 Descriptive Statistics",  # Execute this statement
            # Execute this operation
            "📈 Visualizations",  # Execute this statement
            # Execute this operation
            "🎲 Sampling",  # Execute this statement
            # Execute this operation
            "⚖️ Normalization",  # Execute this statement
            # Execute this operation
            "🔔 Distributions",  # Execute this statement
            # Execute this operation
            "🔗 Fitting & CLT",  # Execute this statement
            # Execute this operation
            "🧪 Hypothesis Testing",  # Execute this statement
        # Execute this operation
        ],  # Execute this statement
        label_visibility="collapsed",   # Hide the label (already shown as heading above)
        # Compute and store the result in key
        key="nav_radio"  # Store result in key
    )

    st.markdown("---")   # Bottom divider

    # Show dataset info in sidebar if a file is loaded
    # Check condition and branch accordingly
    if uploaded_file is not None:  # Check condition
        # Display a green indicator when file is loaded
        # Render formatted markdown text in the Streamlit UI
        st.markdown("### ✅ Dataset Loaded")  # Render formatted markdown text in the Streamlit UI
        # These will be filled in after loading below


# =====================================================================
# DATASET LOADING
# Load the uploaded CSV into a pandas DataFrame and store in session
# =====================================================================

# st.session_state is a dictionary that persists across Streamlit reruns
# We use it to store the loaded DataFrame so it doesn't disappear on interaction

# Check condition and branch accordingly
if "df" not in st.session_state:  # Check condition
    st.session_state["df"] = None   # Initialize to None if not yet loaded

# Check condition and branch accordingly
if uploaded_file is not None:  # Check condition
    try:
        # Read the uploaded file's bytes as a string buffer
        # This is needed because Streamlit's file_uploader returns a BytesIO object
        bytes_data = uploaded_file.read()   # Read raw bytes from the uploaded file

        # Try UTF-8 first, fall back to latin-1 for special characters
        try:
            df_loaded = pd.read_csv(io.BytesIO(bytes_data), encoding="utf-8")   # Parse CSV
        # Handle any error that occurred in the try block
        except UnicodeDecodeError:  # Handle any error from the try block
            df_loaded = pd.read_csv(io.BytesIO(bytes_data), encoding="latin-1")  # Fallback

        # Save the DataFrame into session state so all pages can access it
        # Access persistent state that survives page reruns
        st.session_state["df"] = df_loaded  # Access persistent state across reruns

    # Handle any error that occurred in the try block
    except Exception as e:  # Handle any error from the try block
        # If parsing fails, show a friendly error message
        # Display content in the sidebar panel
        st.sidebar.error(f"Could not read file: {e}")  # Execute this statement

# Retrieve the DataFrame from session state (may be None if no file loaded)
# Pandas data operation
df = st.session_state.get("df", None)  # Store result in df

# Update the sidebar with dataset info if loaded
# Check condition and branch accordingly
if df is not None:  # Check condition
    # Open a context manager (auto-closes when done)
    with st.sidebar:  # Open context manager
        st.markdown(f"- **Rows:** {df.shape[0]}")             # Show row count
        st.markdown(f"- **Columns:** {df.shape[1]}")          # Show column count
        st.markdown(f"- **Numeric cols:** {len(df.select_dtypes(include='number').columns)}")  # Numeric count
        missing_pct = 100 * df.isnull().sum().sum() / (df.shape[0] * df.shape[1])  # % missing
        st.markdown(f"- **Missing values:** {missing_pct:.1f}%")  # Show missing %

    # Render the unit manager in the sidebar so user can define units for all numeric columns
    # These units will appear on chart axes and result displays throughout the dashboard
    render_unit_manager(df)  # Call unit manager — shows text inputs for each numeric column


# =====================================================================
# PAGE ROUTING
# Display the correct module based on sidebar navigation selection
# =====================================================================

# ---- HOME PAGE ----
# Check condition and branch accordingly
if page == "🏠 Home":  # Check condition
    # Display the main dashboard welcome page
    # Display a large page title heading
    st.title("📊 Statistical Analysis Dashboard")  # Display a large page title
    # Render formatted markdown text in the Streamlit UI
    st.markdown("### Welcome to your interactive statistics toolbox")  # Render formatted markdown text in the Streamlit UI
    # Render formatted markdown text in the Streamlit UI
    st.markdown("""  # Render formatted markdown text in the Streamlit UI
    # Execute this operation
    This dashboard provides a complete statistical analysis workflow for any CSV dataset.  # Execute this statement
    # Execute this operation
    Upload a file using the sidebar and navigate through the modules below.  # Execute this statement
    """)

    # Display a feature overview grid using Streamlit columns
    col1, col2, col3 = st.columns(3)   # Three columns for feature cards

    # Open a context manager (auto-closes when done)
    with col1:  # Open context manager
        # Show an informational blue message box
        st.info("**📐 Descriptive Statistics**\nMean, Median, Mode, Variance, Std Dev with formulas")  # Show an informational blue message box
        # Show an informational blue message box
        st.info("**📈 Visualizations**\nHistogram, Boxplot, Scatter, KDE, Spider, Gauge & more")  # Show an informational blue message box
        # Show an informational blue message box
        st.info("**🎲 Sampling**\nRandom, Systematic, Stratified sampling methods")  # Show an informational blue message box

    # Open a context manager (auto-closes when done)
    with col2:  # Open context manager
        # Show an informational blue message box
        st.info("**⚖️ Normalization**\nMin-Max scaling and Z-score standardization")  # Show an informational blue message box
        # Show an informational blue message box
        st.info("**🔔 Distributions**\nNormal, Poisson, Exponential, Binomial, Bernoulli, Uniform")  # Show an informational blue message box
        # Show an informational blue message box
        st.info("**🔗 Fitting & CLT**\nFit your data to distributions + CLT simulation")  # Show an informational blue message box

    # Open a context manager (auto-closes when done)
    with col3:  # Open context manager
        # Show an informational blue message box
        st.info("**🧪 Hypothesis Testing**\nT-test, Z-test, ANOVA, Chi-square with interpretation")  # Show an informational blue message box
        # Show a yellow warning message to the user
        st.warning("**📂 Getting Started**\nUpload a CSV file using the sidebar to begin analysis")  # Show a yellow warning message
        # Show a green success message to the user
        st.success("**✅ All modules are fully interactive and educational**")  # Show a green success message

    # If no file is uploaded, show a demo prompt
    # Check condition and branch accordingly
    if df is None:  # Check condition
        # Render formatted markdown text in the Streamlit UI
        st.markdown("---")  # Render formatted markdown text in the Streamlit UI
        # Render formatted markdown text in the Streamlit UI
        st.markdown("### 🚀 Quick Start")  # Render formatted markdown text in the Streamlit UI
        # Render formatted markdown text in the Streamlit UI
        st.markdown("""  # Render formatted markdown text in the Streamlit UI
        # Execute this operation
        1. Click **Browse files** in the sidebar  # Execute this statement
        # Execute this operation
        2. Upload any CSV dataset (e.g., Iris, Titanic, or your own)  # Execute this statement
        # Execute this operation
        3. Use the sidebar menu to navigate to any module  # Execute this statement
        # Execute this operation
        4. Select columns and click buttons to explore your data  # Execute this statement
        """)
        # Show an informational blue message box
        st.info("💡 Tip: You can download sample datasets from [Kaggle](https://www.kaggle.com/datasets) or use any CSV file.")  # Show an informational blue message box
    else:
        # Show a preview of the loaded dataset
        # Render formatted markdown text in the Streamlit UI
        st.markdown("---")  # Render formatted markdown text in the Streamlit UI
        # Render formatted markdown text in the Streamlit UI
        st.markdown("### 📋 Dataset Preview")  # Render formatted markdown text in the Streamlit UI
        st.dataframe(df.head(10), use_container_width=True)   # Show first 10 rows
        # Show small grey caption text below a widget
        st.caption(f"Showing first 10 of {len(df)} rows · {len(df.columns)} columns")  # Show small grey caption text

        # Show basic data info
        # Open a context manager (auto-closes when done)
        with st.expander("📊 Column Data Types"):  # Open context manager
            # Create a summary of column names and their data types
            # Create a new DataFrame from a dictionary or array
            dtype_df = pd.DataFrame({  # Create DataFrame from dictionary or array
                "Column": df.dtypes.index,                   # Column names
                "Data Type": df.dtypes.values.astype(str),  # Their dtypes as strings
                "Non-Null Count": df.count().values,         # Count of non-null values
                "Null Count": df.isnull().sum().values,      # Count of null values
            # Execute this operation
            })  # Execute this statement
            # Render an interactive data table in the UI
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)  # Render an interactive data table

# ---- STATISTICS PAGE ----
# Alternative condition check
elif page == "📐 Descriptive Statistics":  # Check alternative condition
    if validate_dataframe(df):      # Validate that dataset is loaded
        render_statistics(df)       # Call the statistics module

# ---- VISUALIZATIONS PAGE ----
# Alternative condition check
elif page == "📈 Visualizations":  # Check alternative condition
    if validate_dataframe(df):      # Validate dataset
        render_plots(df)            # Call the plots module

# ---- SAMPLING PAGE ----
# Alternative condition check
elif page == "🎲 Sampling":  # Check alternative condition
    if validate_dataframe(df):      # Validate dataset
        render_sampling(df)         # Call the sampling module

# ---- NORMALIZATION PAGE ----
# Alternative condition check
elif page == "⚖️ Normalization":  # Check alternative condition
    if validate_dataframe(df):      # Validate dataset
        render_normalization(df)    # Call the normalization module

# ---- DISTRIBUTIONS PAGE ----
# Alternative condition check
elif page == "🔔 Distributions":  # Check alternative condition
    # Distributions module doesn't need a dataset — it's theoretical
    render_distributions(df)        # Call distributions module (df may be None here)

# ---- FITTING & CLT PAGE ----
# Alternative condition check
elif page == "🔗 Fitting & CLT":  # Check alternative condition
    if validate_dataframe(df):      # Validate dataset (CLT simulation needs data)
        render_fitting(df)          # Call the fitting and CLT module

# ---- HYPOTHESIS TESTING PAGE ----
# Alternative condition check
elif page == "🧪 Hypothesis Testing":  # Check alternative condition
    if validate_dataframe(df):      # Validate dataset
        render_tests(df)            # Call the hypothesis testing module


# =====================================================================
# FOOTER
# =====================================================================

st.markdown("---")   # Bottom divider
# Render formatted markdown text in the Streamlit UI
st.markdown(  # Render formatted markdown text in the Streamlit UI
    # Compute and store the result in "<div style
    "<div style='text-align:center; color:#999; font-size:0.8rem;'>"  # Store result in "<div style
    # Execute this operation
    "Statistical Dashboard · Built with Python & Streamlit · Educational Use"  # Execute this statement
    # Execute this operation
    "</div>",  # Execute this statement
    unsafe_allow_html=True   # Allow HTML for centered/styled footer text
)
