# =========================
# modules/plots.py
# Visualization module
# Provides Histogram, Boxplot, Scatter, Line, KDE,
# Spider (Radar) chart, and Gauge chart
# =========================

import pandas as pd                  # Import pandas for DataFrame handling
import numpy as np                   # Import numpy for numerical operations
import matplotlib.pyplot as plt      # Import matplotlib for static plots
import seaborn as sns                # Import seaborn for styled statistical plots
import plotly.graph_objects as go    # Import plotly graph objects for interactive charts
import plotly.express as px          # Import plotly express for quick interactive plots
import streamlit as st               # Import streamlit for UI rendering
from utils.helpers import (          # Import shared utilities
    get_numeric_columns,             # To list numeric columns for dropdown
    get_categorical_columns,         # To list categorical columns
    drop_missing,                    # To remove NaN before plotting
    section_header,                  # To display a styled heading
)


def render_plots(df: pd.DataFrame):
    """
    Main function for the Plots module.
    Renders the complete visualization UI inside the Streamlit app.

    Parameters:
        df (pd.DataFrame): The uploaded dataset
    """
    section_header("Visualizations", "📈")  # Display section title

    # Get list of numeric and categorical columns for dropdowns
    numeric_cols = get_numeric_columns(df)   # Only columns with numbers
    all_cols = df.columns.tolist()           # All columns (for line plot x-axis)

    # If there are no numeric columns, warn the user and stop
    if not numeric_cols:
        st.error("No numeric columns found in the dataset.")
        return  # Exit the function — nothing to plot

    # Let the user select which type of plot to create
    plot_type = st.selectbox(
        "Select plot type:",
        [                         # List of available chart types
            "Histogram",
            "Boxplot",
            "Scatter Plot",
            "Line Plot",
            "KDE / Density Plot",
            "Spider / Radar Chart",
            "Gauge Chart",
            "Correlation Heatmap",
            "Violin Plot",
            "Bar Chart",
        ],
        key="plot_type_select"    # Unique key to avoid Streamlit widget conflicts
    )

    # ---- HISTOGRAM ----
    if plot_type == "Histogram":
        col = st.selectbox("Select column:", numeric_cols, key="hist_col")  # Column dropdown
        bins = st.slider("Number of bins:", 5, 100, 20, key="hist_bins")    # Bin count slider
        if st.button("Generate Histogram"):                                  # Trigger on button
            _plot_histogram(df, col, bins)                                   # Call plot function

    # ---- BOXPLOT ----
    elif plot_type == "Boxplot":
        col = st.selectbox("Select column:", numeric_cols, key="box_col")  # Column dropdown
        if st.button("Generate Boxplot"):                                   # Trigger on button
            _plot_boxplot(df, col)                                          # Call plot function

    # ---- SCATTER PLOT ----
    elif plot_type == "Scatter Plot":
        x_col = st.selectbox("X-axis column:", numeric_cols, key="scatter_x")   # X column
        y_col = st.selectbox("Y-axis column:", numeric_cols, key="scatter_y")   # Y column
        if st.button("Generate Scatter Plot"):                                   # Trigger
            _plot_scatter(df, x_col, y_col)                                      # Call function

    # ---- LINE PLOT ----
    elif plot_type == "Line Plot":
        x_col = st.selectbox("X-axis column:", all_cols, key="line_x")          # X axis
        y_col = st.selectbox("Y-axis column:", numeric_cols, key="line_y")      # Y axis
        if st.button("Generate Line Plot"):                                      # Trigger
            _plot_line(df, x_col, y_col)                                         # Call function

    # ---- KDE / DENSITY PLOT ----
    elif plot_type == "KDE / Density Plot":
        col = st.selectbox("Select column:", numeric_cols, key="kde_col")   # Column dropdown
        if st.button("Generate KDE Plot"):                                   # Trigger
            _plot_kde(df, col)                                               # Call function

    # ---- SPIDER / RADAR CHART ----
    elif plot_type == "Spider / Radar Chart":
        selected_cols = st.multiselect(                        # Allow multi-column selection
            "Select numeric columns (min 3):",
            numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))],  # Default to first 5 columns
            key="spider_cols"
        )
        if st.button("Generate Spider Chart"):                  # Trigger on button
            if len(selected_cols) < 3:                          # Spider needs at least 3 axes
                st.warning("Please select at least 3 columns.")
            else:
                _plot_spider(df, selected_cols)                  # Call spider chart function

    # ---- GAUGE CHART ----
    elif plot_type == "Gauge Chart":
        col = st.selectbox("Select column:", numeric_cols, key="gauge_col")   # Column dropdown
        if st.button("Generate Gauge Chart"):                                  # Trigger
            _plot_gauge(df, col)                                               # Call function

    # ---- CORRELATION HEATMAP ----
    elif plot_type == "Correlation Heatmap":
        if st.button("Generate Heatmap"):       # Trigger on button
            _plot_heatmap(df, numeric_cols)     # Call heatmap function

    # ---- VIOLIN PLOT ----
    elif plot_type == "Violin Plot":
        col = st.selectbox("Select column:", numeric_cols, key="violin_col")   # Column dropdown
        if st.button("Generate Violin Plot"):                                   # Trigger
            _plot_violin(df, col)                                               # Call function

    # ---- BAR CHART ----
    elif plot_type == "Bar Chart":
        cat_cols = get_categorical_columns(df)   # Get categorical columns for grouping
        if not cat_cols:
            st.warning("No categorical columns found for bar chart grouping.")
        else:
            cat_col = st.selectbox("Category column:", cat_cols, key="bar_cat")        # Category
            num_col = st.selectbox("Numeric column:", numeric_cols, key="bar_num")     # Value
            if st.button("Generate Bar Chart"):                                         # Trigger
                _plot_bar(df, cat_col, num_col)                                         # Call


# =====================================================================
# Individual plot functions below
# Each function handles one chart type with full comments
# =====================================================================

def _plot_histogram(df: pd.DataFrame, col: str, bins: int):
    """Create and display a histogram using matplotlib and seaborn."""
    fig, ax = plt.subplots(figsize=(9, 5))                  # Create figure and axis objects
    series = drop_missing(df[col])                          # Remove NaN values
    sns.histplot(series, bins=bins, kde=True, ax=ax,        # Plot histogram with KDE overlay
                 color="#4C72B0", edgecolor="white")        # Use blue bars with white borders
    ax.set_title(f"Histogram of {col}", fontsize=14)        # Set chart title
    ax.set_xlabel(col)                                       # Label x-axis with column name
    ax.set_ylabel("Frequency")                              # Label y-axis
    ax.grid(axis="y", alpha=0.3)                            # Add faint horizontal grid lines
    st.pyplot(fig)                                          # Render the figure in Streamlit
    plt.close(fig)                                          # Close figure to free memory

    # Show an explanation of the chart
    with st.expander("📖 What is a Histogram?"):
        st.write("""
        A histogram divides the data into equal-width intervals (bins) and 
        counts how many values fall in each bin. The KDE line shows the 
        smooth probability density estimate.
        """)


def _plot_boxplot(df: pd.DataFrame, col: str):
    """Create and display a boxplot showing quartiles and outliers."""
    fig, ax = plt.subplots(figsize=(7, 5))                         # Create figure
    series = drop_missing(df[col])                                 # Clean data
    sns.boxplot(y=series, ax=ax, color="#55A868", width=0.4)       # Draw boxplot in green
    ax.set_title(f"Boxplot of {col}", fontsize=14)                 # Set chart title
    ax.set_ylabel(col)                                             # Label y-axis
    ax.grid(axis="y", alpha=0.3)                                   # Faint grid
    st.pyplot(fig)                                                 # Render in Streamlit
    plt.close(fig)                                                 # Free memory

    # Show five-number summary below the plot
    with st.expander("📖 Boxplot Explanation"):
        st.write(f"""
        **Five-Number Summary for '{col}':**
        - Min: {series.min():.4f}
        - Q1 (25th percentile): {series.quantile(0.25):.4f}
        - Median (Q2): {series.median():.4f}
        - Q3 (75th percentile): {series.quantile(0.75):.4f}
        - Max: {series.max():.4f}
        
        Points outside 1.5 × IQR are shown as dots (outliers).
        """)


def _plot_scatter(df: pd.DataFrame, x_col: str, y_col: str):
    """Create and display an interactive scatter plot using Plotly."""
    # Drop rows where either selected column has NaN
    clean_df = df[[x_col, y_col]].dropna()

    # Create a plotly scatter figure
    fig = px.scatter(
        clean_df,              # Use cleaned DataFrame
        x=x_col,               # Column for x-axis
        y=y_col,               # Column for y-axis
        title=f"Scatter Plot: {x_col} vs {y_col}",  # Chart title
        trendline="ols",       # Add OLS (ordinary least squares) regression line
        template="plotly_white",  # Clean white background theme
        color_discrete_sequence=["#4C72B0"],  # Use consistent blue color
    )
    st.plotly_chart(fig, use_container_width=True)  # Render interactive chart


def _plot_line(df: pd.DataFrame, x_col: str, y_col: str):
    """Create and display a line plot using Plotly."""
    clean_df = df[[x_col, y_col]].dropna()  # Remove rows with NaN in either column

    # Create an interactive line chart
    fig = px.line(
        clean_df,              # Data source
        x=x_col,               # X-axis (e.g., time or index)
        y=y_col,               # Y-axis (numeric values)
        title=f"Line Plot: {y_col} over {x_col}",  # Title
        template="plotly_white",                    # White background
        markers=True,                               # Show data point markers on the line
    )
    st.plotly_chart(fig, use_container_width=True)  # Render in Streamlit


def _plot_kde(df: pd.DataFrame, col: str):
    """Create and display a KDE (Kernel Density Estimate) plot."""
    fig, ax = plt.subplots(figsize=(9, 5))            # Create figure
    series = drop_missing(df[col])                    # Clean data
    sns.kdeplot(series, ax=ax, fill=True,             # Draw filled KDE curve
                color="#C44E52", alpha=0.6)           # Red fill with transparency
    ax.set_title(f"KDE Density Plot of {col}", fontsize=14)  # Title
    ax.set_xlabel(col)                                # x-axis label
    ax.set_ylabel("Density")                          # y-axis label
    ax.grid(alpha=0.3)                                # Faint grid
    st.pyplot(fig)                                    # Render
    plt.close(fig)                                    # Free memory

    with st.expander("📖 What is a KDE Plot?"):
        st.write("""
        A KDE (Kernel Density Estimate) plot shows the probability distribution 
        of a variable. Unlike a histogram, it is smooth. The area under the curve 
        sums to 1, showing the relative likelihood of each value.
        """)


def _plot_spider(df: pd.DataFrame, cols: list):
    """Create and display a spider/radar chart comparing column averages."""
    # Compute the mean of each selected numeric column (one value per axis)
    means = [drop_missing(df[c]).mean() for c in cols]  # List of mean values

    # Spider chart requires closing the polygon — repeat the first value at the end
    values = means + [means[0]]          # Append the first value to close the shape
    labels = cols + [cols[0]]            # Append the first label to close the labels

    # Create a plotly polar (radar) chart
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,           # Radial values (distances from center)
        theta=labels,       # Axis labels (column names)
        fill="toself",      # Fill the polygon area
        name="Column Means",  # Legend label
        line_color="#4C72B0",  # Line color
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),  # Show the radial (value) axis
        title="Spider / Radar Chart of Column Means",  # Chart title
        template="plotly_white",                        # Clean theme
    )
    st.plotly_chart(fig, use_container_width=True)  # Render in Streamlit

    with st.expander("📖 What is a Spider Chart?"):
        st.write("""
        A Spider (Radar) chart displays multiple variables as axes radiating from 
        a central point. Each axis shows the normalized or raw mean of that column. 
        Useful for comparing profiles across multiple dimensions.
        """)


def _plot_gauge(df: pd.DataFrame, col: str):
    """Create and display a gauge chart showing the mean value of a column."""
    series = drop_missing(df[col])   # Clean column data
    mean_val = series.mean()         # Compute the mean to display on the gauge
    min_val = series.min()           # Minimum value — used as gauge lower bound
    max_val = series.max()           # Maximum value — used as gauge upper bound

    # Build an interactive gauge using plotly Indicator
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",   # Show gauge dial, numeric value, and change from min
        value=mean_val,              # The value to display on the gauge
        title={"text": f"Mean of {col}"},  # Title inside the gauge
        delta={"reference": min_val},      # Show delta compared to the minimum
        gauge={
            "axis": {"range": [min_val, max_val]},  # Full range from min to max
            "bar": {"color": "#4C72B0"},             # Blue gauge bar
            "steps": [                              # Color zones on the gauge
                {"range": [min_val, (min_val + max_val) / 2], "color": "#d9eaf7"},  # Low zone
                {"range": [(min_val + max_val) / 2, max_val], "color": "#a8c8e8"},  # High zone
            ],
        }
    ))
    fig.update_layout(template="plotly_white")   # Clean background
    st.plotly_chart(fig, use_container_width=True)  # Render in Streamlit


def _plot_heatmap(df: pd.DataFrame, numeric_cols: list):
    """Create and display a correlation heatmap for all numeric columns."""
    corr_matrix = df[numeric_cols].corr()           # Compute pairwise Pearson correlations

    fig, ax = plt.subplots(figsize=(10, 7))         # Create figure — larger for many columns
    sns.heatmap(
        corr_matrix,          # Input: correlation matrix
        annot=True,           # Show numeric values inside each cell
        fmt=".2f",            # Format to 2 decimal places
        cmap="coolwarm",      # Red = positive, blue = negative correlation
        linewidths=0.5,       # Thin grid lines between cells
        ax=ax,                # Draw on our axis
        vmin=-1, vmax=1,      # Fix color scale from -1 to +1
    )
    ax.set_title("Correlation Heatmap", fontsize=14)  # Chart title
    st.pyplot(fig)                                    # Render
    plt.close(fig)                                    # Free memory


def _plot_violin(df: pd.DataFrame, col: str):
    """Create and display a violin plot showing distribution shape."""
    fig, ax = plt.subplots(figsize=(7, 5))                       # Create figure
    series = drop_missing(df[col])                               # Clean data
    sns.violinplot(y=series, ax=ax, color="#8172B2", inner="box")  # Purple violin with inner box
    ax.set_title(f"Violin Plot of {col}", fontsize=14)           # Title
    ax.set_ylabel(col)                                           # y-axis label
    ax.grid(axis="y", alpha=0.3)                                 # Faint grid
    st.pyplot(fig)                                               # Render
    plt.close(fig)                                               # Free memory

    with st.expander("📖 What is a Violin Plot?"):
        st.write("""
        A Violin plot combines a boxplot and a KDE. The width of the violin at 
        each value shows the density (how common that value is). The inner box 
        shows the median and interquartile range.
        """)


def _plot_bar(df: pd.DataFrame, cat_col: str, num_col: str):
    """Create and display a bar chart of a numeric column grouped by a category."""
    # Group by the categorical column and compute the mean of the numeric column
    grouped = df.groupby(cat_col)[num_col].mean().reset_index()

    # Create an interactive bar chart with plotly
    fig = px.bar(
        grouped,               # Data source — grouped means
        x=cat_col,             # X-axis: category names
        y=num_col,             # Y-axis: mean values
        title=f"Mean {num_col} by {cat_col}",  # Chart title
        template="plotly_white",               # Clean theme
        color=cat_col,                         # Different color per category
        text_auto=".2f",                       # Show values on top of bars
    )
    st.plotly_chart(fig, use_container_width=True)  # Render in Streamlit
