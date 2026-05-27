# =========================
# modules/plots.py
# Visualization module
# Provides Histogram, Boxplot, Scatter, Line, KDE,
# Spider (Radar) chart, and Gauge chart
# =========================

import pandas as pd                  # Import pandas for DataFrame handling
import numpy as np                   # Import numpy for numerical operations
import re                            # Import re for regular expressions used in column name parsing
import matplotlib.pyplot as plt      # Import matplotlib for static plots
import seaborn as sns                # Import seaborn for styled statistical plots
import plotly.graph_objects as go    # Import plotly graph objects for interactive charts
import plotly.express as px          # Import plotly express for quick interactive plots
import streamlit as st               # Import streamlit for UI rendering
from utils.helpers import (          # Import shared utility functions
    get_numeric_columns,             # To list numeric columns for dropdown
    get_categorical_columns,         # To list categorical columns
    drop_missing,                    # To remove NaN before plotting
    section_header,                  # To display a styled heading
    label_with_unit,                 # Build axis label with unit e.g. "Salary ($)"
    freq_label,                      # Standard Y-axis label for histograms
    density_label,                   # Standard Y-axis label for KDE/density plots
    probability_label,               # Standard Y-axis label for PMF plots
    cdf_label,                       # Standard Y-axis label for CDF plots
    render_inline_unit_input,        # Inline unit input widget next to column selector
)


# Define the render_plots function
def render_plots(df: pd.DataFrame):  # Define the render_plots function
    """
    # Execute this operation
    Main function for the Plots module.  # Execute this statement
    # Execute this operation
    Renders the complete visualization UI inside the Streamlit app.  # Execute this statement

    # Execute this operation
    Parameters:  # Execute this statement
        # Create a new DataFrame from a dictionary or array
        df (pd.DataFrame): The uploaded dataset  # Create DataFrame from dictionary or array
    """
    section_header("Visualizations", "📈")  # Display section title

    # Get list of numeric and categorical columns for dropdowns
    numeric_cols = get_numeric_columns(df)   # Only columns with numbers
    all_cols = df.columns.tolist()           # All columns (for line plot x-axis)

    # If there are no numeric columns, warn the user and stop
    # Check condition and branch accordingly
    if not numeric_cols:  # Check condition
        # Show a red error message to the user
        st.error("No numeric columns found in the dataset.")  # Show a red error message
        return  # Exit the function — nothing to plot

    # Let the user select which type of plot to create
    # Compute and store the result in plot_type
    plot_type = st.selectbox(  # Store result in plot_type
        # Execute this operation
        "Select plot type:",  # Execute this statement
        [                         # List of available chart types
            # Execute this operation
            "Histogram",  # Execute this statement
            # Execute this operation
            "Boxplot",  # Execute this statement
            # Execute this operation
            "Scatter Plot",  # Execute this statement
            # Execute this operation
            "Line Plot",  # Execute this statement
            # Execute this operation
            "KDE / Density Plot",  # Execute this statement
            # Execute this operation
            "Spider / Radar Chart",  # Execute this statement
            # Execute this operation
            "Gauge Chart",  # Execute this statement
            # Execute this operation
            "Correlation Heatmap",  # Execute this statement
            # Execute this operation
            "Violin Plot",  # Execute this statement
            # Execute this operation
            "Bar Chart",  # Execute this statement
        # Execute this operation
        ],  # Execute this statement
        key="plot_type_select"    # Unique key to avoid Streamlit widget conflicts
    )

    # ---- HISTOGRAM ----
    # Check condition and branch accordingly
    if plot_type == "Histogram":  # Check condition
        col = st.selectbox("Select column:", numeric_cols, key="hist_col")  # Column dropdown
        render_inline_unit_input(col, "hist")  # Ask user for the unit of this column
        bins = st.slider("Number of bins:", 5, 100, 20, key="hist_bins")    # Bin count slider
        if st.button("Generate Histogram"):                                  # Trigger on button
            _plot_histogram(df, col, bins)                                   # Call plot function

    # ---- BOXPLOT ----
    # Alternative condition check
    elif plot_type == "Boxplot":  # Check alternative condition
        col = st.selectbox("Select column:", numeric_cols, key="box_col")  # Column dropdown
        render_inline_unit_input(col, "box")  # Ask user for the unit of this column
        if st.button("Generate Boxplot"):                                   # Trigger on button
            _plot_boxplot(df, col)                                          # Call plot function

    # ---- SCATTER PLOT ----
    # Alternative condition check
    elif plot_type == "Scatter Plot":  # Check alternative condition
        x_col = st.selectbox("X-axis column:", numeric_cols, key="scatter_x")   # X column
        render_inline_unit_input(x_col, "scatter_x")  # Unit for X axis
        y_col = st.selectbox("Y-axis column:", numeric_cols, key="scatter_y")   # Y column
        render_inline_unit_input(y_col, "scatter_y")  # Unit for Y axis
        if st.button("Generate Scatter Plot"):                                   # Trigger
            _plot_scatter(df, x_col, y_col)                                      # Call function

    # ---- LINE PLOT ----
    # Alternative condition check
    elif plot_type == "Line Plot":  # Check alternative condition
        x_col = st.selectbox("X-axis column:", all_cols, key="line_x")          # X axis
        render_inline_unit_input(x_col, "line_x")  # Unit for x axis
        y_col = st.selectbox("Y-axis column:", numeric_cols, key="line_y")      # Y axis
        render_inline_unit_input(y_col, "line_y")  # Unit for y axis
        if st.button("Generate Line Plot"):                                      # Trigger
            _plot_line(df, x_col, y_col)                                         # Call function

    # ---- KDE / DENSITY PLOT ----
    # Alternative condition check
    elif plot_type == "KDE / Density Plot":  # Check alternative condition
        col = st.selectbox("Select column:", numeric_cols, key="kde_col")   # Column dropdown
        render_inline_unit_input(col, "kde")  # Ask user for the unit of this column
        if st.button("Generate KDE Plot"):                                   # Trigger
            _plot_kde(df, col)                                               # Call function

    # ---- SPIDER / RADAR CHART ----
    # Alternative condition check
    elif plot_type == "Spider / Radar Chart":  # Check alternative condition
        selected_cols = st.multiselect(                        # Allow multi-column selection
            # Execute this operation
            "Select numeric columns (min 3):",  # Execute this statement
            # Execute this operation
            numeric_cols,  # Execute this statement
            default=numeric_cols[:min(5, len(numeric_cols))],  # Default to first 5 columns
            # Compute and store the result in key
            key="spider_cols"  # Store result in key
        )
        if st.button("Generate Spider Chart"):                  # Trigger on button
            if len(selected_cols) < 3:                          # Spider needs at least 3 axes
                # Show a yellow warning message to the user
                st.warning("Please select at least 3 columns.")  # Show a yellow warning message
            else:
                _plot_spider(df, selected_cols)                  # Call spider chart function

    # ---- GAUGE CHART ----
    # Alternative condition check
    elif plot_type == "Gauge Chart":  # Check alternative condition
        col = st.selectbox("Select column:", numeric_cols, key="gauge_col")   # Column dropdown
        render_inline_unit_input(col, "gauge")  # Unit input for gauge axis
        if st.button("Generate Gauge Chart"):                                  # Trigger
            _plot_gauge(df, col)                                               # Call function

    # ---- CORRELATION HEATMAP ----
    # Alternative condition check
    elif plot_type == "Correlation Heatmap":  # Check alternative condition
        if st.button("Generate Heatmap"):       # Trigger on button
            _plot_heatmap(df, numeric_cols)     # Call heatmap function

    # ---- VIOLIN PLOT ----
    # Alternative condition check
    elif plot_type == "Violin Plot":  # Check alternative condition
        col = st.selectbox("Select column:", numeric_cols, key="violin_col")   # Column dropdown
        render_inline_unit_input(col, "violin")  # Ask user for the unit of this column
        if st.button("Generate Violin Plot"):                                   # Trigger
            _plot_violin(df, col)                                               # Call function

    # ---- BAR CHART ----
    # Alternative condition check
    elif plot_type == "Bar Chart":  # Check alternative condition
        cat_cols = get_categorical_columns(df)   # Get categorical columns for grouping
        # Check condition and branch accordingly
        if not cat_cols:  # Check condition
            # Show a yellow warning message to the user
            st.warning("No categorical columns found for bar chart grouping.")  # Show a yellow warning message
        else:
            cat_col = st.selectbox("Category column:", cat_cols, key="bar_cat")        # Category
            num_col = st.selectbox("Numeric column:", numeric_cols, key="bar_num")     # Value
            render_inline_unit_input(num_col, "bar_num")  # Unit for bar chart y-axis
            if st.button("Generate Bar Chart"):                                         # Trigger
                _plot_bar(df, cat_col, num_col)                                         # Call


# =====================================================================
# Individual plot functions below
# Each function handles one chart type with full comments
# =====================================================================

# Define the _plot_histogram function
def _plot_histogram(df: pd.DataFrame, col: str, bins: int):  # Define the _plot_histogram function
    """Create and display a histogram using matplotlib and seaborn."""
    fig, ax = plt.subplots(figsize=(9, 5))                  # Create figure and axis objects
    series = drop_missing(df[col])                          # Remove NaN values
    sns.histplot(series, bins=bins, kde=True, ax=ax,        # Plot histogram with KDE overlay
                 color="#4C72B0", edgecolor="white")        # Use blue bars with white borders
    ax.set_title(f"Histogram of {col}", fontsize=14)        # Set chart title
    ax.set_xlabel(label_with_unit(col))                      # Label x-axis with column name and unit
    ax.set_ylabel(freq_label())                              # Label y-axis with standard frequency label
    ax.grid(axis="y", alpha=0.3)                            # Add faint horizontal grid lines
    st.pyplot(fig)                                          # Render the figure in Streamlit
    plt.close(fig)                                          # Close figure to free memory

    # Show an explanation of the chart
    # Open a context manager (auto-closes when done)
    with st.expander("📖 What is a Histogram?"):  # Open context manager
        # Display text or data in the Streamlit UI
        st.write("""  # Display text or data in the Streamlit UI
        # Execute this operation
        A histogram divides the data into equal-width intervals (bins) and   # Execute this statement
        # Execute this operation
        counts how many values fall in each bin. The KDE line shows the   # Execute this statement
        # Execute this operation
        smooth probability density estimate.  # Execute this statement
        """)


# Define the _plot_boxplot function
def _plot_boxplot(df: pd.DataFrame, col: str):  # Define the _plot_boxplot function
    """Create and display a boxplot showing quartiles and outliers."""
    fig, ax = plt.subplots(figsize=(7, 5))                         # Create figure
    series = drop_missing(df[col])                                 # Clean data
    sns.boxplot(y=series, ax=ax, color="#55A868", width=0.4)       # Draw boxplot in green
    ax.set_title(f"Boxplot of {col}", fontsize=14)                 # Set chart title
    ax.set_ylabel(label_with_unit(col))                           # Label y-axis with column name and unit
    ax.grid(axis="y", alpha=0.3)                                   # Faint grid
    st.pyplot(fig)                                                 # Render in Streamlit
    plt.close(fig)                                                 # Free memory

    # Show five-number summary below the plot
    # Open a context manager (auto-closes when done)
    with st.expander("📖 Boxplot Explanation"):  # Open context manager
        # Display text or data in the Streamlit UI
        st.write(f"""  # Display text or data in the Streamlit UI
        # Execute this operation
        **Five-Number Summary for '{col}':**  # Execute this statement
        # Find the smallest value in the column
        - Min: {series.min():.4f}  # Execute this statement
        # Pandas data operation
        - Q1 (25th percentile): {series.quantile(0.25):.4f}  # Execute this statement
        # Find the median (middle) value of the column
        - Median (Q2): {series.median():.4f}  # Execute this statement
        # Pandas data operation
        - Q3 (75th percentile): {series.quantile(0.75):.4f}  # Execute this statement
        # Find the largest value in the column
        - Max: {series.max():.4f}  # Execute this statement
        
        # Execute this operation
        Points outside 1.5 × IQR are shown as dots (outliers).  # Execute this statement
        """)


# Define the _plot_scatter function
def _plot_scatter(df: pd.DataFrame, x_col: str, y_col: str):  # Define the _plot_scatter function
    """Create and display an interactive scatter plot using Plotly."""
    # Drop rows where either selected column has NaN
    # Remove rows with missing (NaN) values
    clean_df = df[[x_col, y_col]].dropna()  # Remove rows with NaN values

    # Create a plotly scatter figure
    # Draw scatter plot points
    fig = px.scatter(  # Store result in fig
        clean_df,              # Use cleaned DataFrame
        x=x_col,               # Column for x-axis
        y=y_col,               # Column for y-axis
        title=f"Scatter Plot: {x_col} vs {y_col}",  # Chart title
        trendline="ols",       # Add OLS (ordinary least squares) regression line
        template="plotly_white",  # Clean white background theme
        color_discrete_sequence=["#4C72B0"],  # Use consistent blue color
    )
    st.plotly_chart(fig, use_container_width=True)  # Render interactive chart


# Define the _plot_line function
def _plot_line(df: pd.DataFrame, x_col: str, y_col: str):  # Define the _plot_line function
    """
    Create a line plot with smart handling based on the X column type.

    THREE MODES depending on X column:

    1. TIME / SEQUENTIAL X (Date, Year, Epoch, OrderID...):
       → Connect raw data points in order. Classic line plot.

    2. NUMERIC X (Age, Years_Experience, Score...):
       → Raw data connecting is meaningless (chaotic tangle).
       → Instead: compute MEAN of Y for each unique X value, then plot that.
       → This shows the trend "how does average Y change as X increases?"
       → Explains what was done so the user understands the chart.

    3. CATEGORICAL X (Department, Gender...):
       → Compute mean of Y per category and plot as a line.
       → Also explains the aggregation clearly.
    """
    import pandas as pd  # Import pandas for dtype checks and aggregation

    # Remove rows where either column is missing
    clean_df = df[[x_col, y_col]].dropna()  # Keep only complete pairs

    if len(clean_df) < 2:  # Need at least 2 points to draw a line
        st.error("Not enough data to draw a line plot (need at least 2 non-missing rows).")
        return  # Exit early

    # Detect X column type to decide how to handle it
    is_datetime = pd.api.types.is_datetime64_any_dtype(df[x_col])  # True if X is date/time

    # Try to detect if string column is actually dates
    looks_like_datetime = False  # Default: assume not datetime
    if not is_datetime and df[x_col].dtype == object:  # Only try if text column
        try:
            pd.to_datetime(clean_df[x_col].head(10), infer_datetime_format=True)  # Attempt date parse
            looks_like_datetime = True  # Parsing worked — treat as datetime
        except Exception:
            looks_like_datetime = False  # Not a date column

    # Check if column name contains time/sequence keywords (exact word match)
    time_keywords = [  # Keywords that strongly suggest sequential/time data
        "date", "time", "year", "month", "day", "week", "quarter",
        "period", "hour", "minute", "second", "index", "sequence",
        "step", "iteration", "epoch", "round", "order", "rank"
    ]
    col_words = re.split(r'[_\s\-]+', x_col.lower())  # Split name into words e.g. "Order_Date" → ["order","date"]
    name_suggests_time = any(kw in col_words for kw in time_keywords)  # True if any word matches

    # Check if numeric column is a true sequential index (all unique, evenly spaced)
    is_numeric_x = pd.api.types.is_numeric_dtype(df[x_col])  # True if X is a number
    looks_like_sequence = False  # Default: not a sequence
    if is_numeric_x and len(clean_df) > 1:  # Only check numeric columns with data
        all_unique = clean_df[x_col].nunique() == len(clean_df)  # All values must be different
        if all_unique:  # Only sequential if no repeated values
            sorted_vals = clean_df[x_col].sort_values().reset_index(drop=True)  # Sort values
            diffs = sorted_vals.diff().dropna()  # Gaps between consecutive values
            looks_like_sequence = diffs.std() == 0  # True only if all gaps are exactly equal

    is_categorical_x = df[x_col].dtype == object or str(df[x_col].dtype) == 'category'  # True if text/category

    # ---- MODE 1: TIME / SEQUENTIAL — plot raw data ----
    if is_datetime or looks_like_datetime or name_suggests_time or looks_like_sequence:

        plot_df = clean_df.sort_values(by=x_col)  # Sort by X so line flows correctly left to right

        fig = px.line(  # Build Plotly line chart
            plot_df,                                    # Sorted clean data
            x=x_col,                                    # X axis: time or sequence column
            y=y_col,                                    # Y axis: numeric values
            title=f"Line Plot: {y_col} over {x_col}",  # Chart title
            template="plotly_white",                    # Clean white background
            markers=True,                               # Show dots at each point
        )
        st.plotly_chart(fig, use_container_width=True)  # Render the interactive chart

        st.caption(f"Showing {len(plot_df)} data points connected in order of '{x_col}'.")  # Explain what's shown

    # ---- MODE 2: REGULAR NUMERIC X — aggregate to mean ----
    elif is_numeric_x:

        # Explain to the user WHY we are aggregating
        st.info(f"""
        ℹ️ **'{x_col}' is a numeric measurement, not a time or sequence column.**

        Connecting {len(clean_df)} individual data points with a line would produce a
        chaotic, meaningless chart (each point is a separate person/record with no order).

        **Instead, this chart shows:** the **mean of '{y_col}'** for each unique value of '{x_col}'.
        This reveals the trend — how does average {y_col} change as {x_col} increases?
        """)

        # Compute mean of Y grouped by each unique value of X
        agg_df = (
            clean_df.groupby(x_col)[y_col]   # Group rows by X value
            .agg(                             # Apply multiple aggregations
                Mean=('mean'),                # Average Y for this X value
                Count=('count'),              # How many rows had this X value
                StdDev=('std'),               # Spread of Y values at this X
            )
            .reset_index()                    # Make X column accessible again
            .sort_values(by=x_col)            # Sort by X for left-to-right line
        )

        # Build the line chart on aggregated means
        fig = px.line(  # Plotly line chart
            agg_df,                                                        # Aggregated data
            x=x_col,                                                       # X: the grouping variable
            y="Mean",                                                      # Y: mean of Y column per X group
            title=f"Mean {y_col} by {x_col}",                             # Descriptive title
            template="plotly_white",                                       # Clean theme
            markers=True,                                                  # Show dots at each group
            labels={"Mean": f"Mean of {y_col}"},                          # Rename Y axis label
        )

        # Add shaded error band (±1 std dev) to show spread at each X value
        upper = agg_df["Mean"] + agg_df["StdDev"].fillna(0)  # Upper bound = mean + std
        lower = agg_df["Mean"] - agg_df["StdDev"].fillna(0)  # Lower bound = mean - std

        fig.add_scatter(  # Add the upper boundary of the std dev band
            x=agg_df[x_col].tolist() + agg_df[x_col].tolist()[::-1],  # X goes forward then backward
            y=upper.tolist() + lower.tolist()[::-1],                   # Y traces upper then lower bound
            fill="toself",           # Fill the enclosed area
            fillcolor="rgba(76,114,176,0.15)",  # Light blue fill for the band
            line=dict(color="rgba(0,0,0,0)"),   # Invisible border line for the band
            name="±1 Std Dev",       # Legend label
            showlegend=True,         # Show in legend
        )

        st.plotly_chart(fig, use_container_width=True)  # Render the chart

        # Show the aggregation table so user can see exact numbers
        with st.expander(f"📋 Aggregated data used for this chart"):  # Expandable table
            display_df = agg_df.copy()  # Copy so we don't modify the original
            display_df.columns = [x_col, f"Mean {y_col}", "Count (rows)", f"Std Dev {y_col}"]  # Rename columns
            display_df = display_df.round(4)  # Round all numbers to 4 decimal places
            st.dataframe(display_df, use_container_width=True, hide_index=True)  # Show table
            st.caption(f"Each row = all records with that '{x_col}' value. Mean is plotted as the line.")

    # ---- MODE 3: CATEGORICAL X — aggregate to mean per category ----
    elif is_categorical_x:

        st.info(f"""
        ℹ️ **'{x_col}' is a categorical column.**
        Showing the **mean of '{y_col}'** for each category.
        """)

        # Compute mean Y per category
        agg_df = (
            clean_df.groupby(x_col)[y_col]  # Group by category
            .mean()                          # Compute mean Y per category
            .reset_index()                   # Restore X as a column
            .sort_values(by=y_col)           # Sort by value for a meaningful left-to-right trend
        )

        fig = px.line(  # Build line chart on category means
            agg_df,
            x=x_col,
            y=y_col,
            title=f"Mean {y_col} by {x_col}",
            template="plotly_white",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)  # Render

    else:
        # Fallback: unknown column type — just inform the user
        st.error(f"Cannot determine how to plot '{x_col}' on a line plot. Try Scatter Plot instead.")
        return  # Exit without plotting


# Define the _plot_kde function
def _plot_kde(df: pd.DataFrame, col: str):  # Define the _plot_kde function
    """Create and display a KDE (Kernel Density Estimate) plot."""
    fig, ax = plt.subplots(figsize=(9, 5))            # Create figure
    series = drop_missing(df[col])                    # Clean data
    sns.kdeplot(series, ax=ax, fill=True,             # Draw filled KDE curve
                color="#C44E52", alpha=0.6)           # Red fill with transparency
    ax.set_title(f"KDE Density Plot of {col}", fontsize=14)  # Title
    ax.set_xlabel(label_with_unit(col))               # x-axis with unit
    ax.set_ylabel(density_label())                    # y-axis with standard density label
    ax.grid(alpha=0.3)                                # Faint grid
    st.pyplot(fig)                                    # Render
    plt.close(fig)                                    # Free memory

    # Open a context manager (auto-closes when done)
    with st.expander("📖 What is a KDE Plot?"):  # Open context manager
        # Display text or data in the Streamlit UI
        st.write("""  # Display text or data in the Streamlit UI
        # Execute this operation
        A KDE (Kernel Density Estimate) plot shows the probability distribution   # Execute this statement
        # Execute this operation
        of a variable. Unlike a histogram, it is smooth. The area under the curve   # Execute this statement
        # Execute this operation
        sums to 1, showing the relative likelihood of each value.  # Execute this statement
        """)


# Define the _plot_spider function
def _plot_spider(df: pd.DataFrame, cols: list):  # Define the _plot_spider function
    """Create and display a spider/radar chart comparing column averages."""
    # Compute the mean of each selected numeric column (one value per axis)
    means = [drop_missing(df[c]).mean() for c in cols]  # List of mean values

    # Spider chart requires closing the polygon — repeat the first value at the end
    values = means + [means[0]]          # Append the first value to close the shape
    labels = cols + [cols[0]]            # Append the first label to close the labels

    # Create a plotly polar (radar) chart
    # Matplotlib or Plotly figure/axis operation
    fig = go.Figure()  # Store result in fig
    # Add a new data trace (line, bar, etc.) to the figure
    fig.add_trace(go.Scatterpolar(  # Execute this statement
        r=values,           # Radial values (distances from center)
        theta=labels,       # Axis labels (column names)
        fill="toself",      # Fill the polygon area
        name="Column Means",  # Legend label
        line_color="#4C72B0",  # Line color
    # Execute this operation
    ))  # Execute this statement
    # Update the Plotly figure layout settings
    fig.update_layout(  # Execute this statement
        polar=dict(radialaxis=dict(visible=True)),  # Show the radial (value) axis
        title="Spider / Radar Chart of Column Means",  # Chart title
        template="plotly_white",                        # Clean theme
    )
    st.plotly_chart(fig, use_container_width=True)  # Render in Streamlit

    # Open a context manager (auto-closes when done)
    with st.expander("📖 What is a Spider Chart?"):  # Open context manager
        # Display text or data in the Streamlit UI
        st.write("""  # Display text or data in the Streamlit UI
        # Execute this operation
        A Spider (Radar) chart displays multiple variables as axes radiating from   # Execute this statement
        # Execute this operation
        a central point. Each axis shows the normalized or raw mean of that column.   # Execute this statement
        # Execute this operation
        Useful for comparing profiles across multiple dimensions.  # Execute this statement
        """)


# Define the _plot_gauge function
def _plot_gauge(df: pd.DataFrame, col: str):  # Define the _plot_gauge function
    """Create and display a gauge chart showing the mean value of a column."""
    series = drop_missing(df[col])   # Clean column data
    mean_val = series.mean()         # Compute the mean to display on the gauge
    min_val = series.min()           # Minimum value — used as gauge lower bound
    max_val = series.max()           # Maximum value — used as gauge upper bound

    # Build an interactive gauge using plotly Indicator
    # Matplotlib or Plotly figure/axis operation
    fig = go.Figure(go.Indicator(  # Store result in fig
        mode="gauge+number+delta",   # Show gauge dial, numeric value, and change from min
        value=mean_val,              # The value to display on the gauge
        title={"text": f"Mean of {col}"},  # Title inside the gauge
        delta={"reference": min_val},      # Show delta compared to the minimum
        # Compute and store the result in gauge
        gauge={  # Store result in gauge
            "axis": {"range": [min_val, max_val]},  # Full range from min to max
            "bar": {"color": "#4C72B0"},             # Blue gauge bar
            "steps": [                              # Color zones on the gauge
                {"range": [min_val, (min_val + max_val) / 2], "color": "#d9eaf7"},  # Low zone
                {"range": [(min_val + max_val) / 2, max_val], "color": "#a8c8e8"},  # High zone
            # Execute this operation
            ],  # Execute this statement
        }
    # Execute this operation
    ))  # Execute this statement
    fig.update_layout(template="plotly_white")   # Clean background
    st.plotly_chart(fig, use_container_width=True)  # Render in Streamlit


# Define the _plot_heatmap function
def _plot_heatmap(df: pd.DataFrame, numeric_cols: list):  # Define the _plot_heatmap function
    """Create and display a correlation heatmap for all numeric columns."""
    corr_matrix = df[numeric_cols].corr()           # Compute pairwise Pearson correlations

    fig, ax = plt.subplots(figsize=(10, 7))         # Create figure — larger for many columns
    # Execute this operation
    sns.heatmap(  # Execute this statement
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


# Define the _plot_violin function
def _plot_violin(df: pd.DataFrame, col: str):  # Define the _plot_violin function
    """Create and display a violin plot showing distribution shape."""
    fig, ax = plt.subplots(figsize=(7, 5))                       # Create figure
    series = drop_missing(df[col])                               # Clean data
    sns.violinplot(y=series, ax=ax, color="#8172B2", inner="box")  # Purple violin with inner box
    ax.set_title(f"Violin Plot of {col}", fontsize=14)           # Title
    ax.set_ylabel(label_with_unit(col))                         # y-axis with unit
    ax.grid(axis="y", alpha=0.3)                                 # Faint grid
    st.pyplot(fig)                                               # Render
    plt.close(fig)                                               # Free memory

    # Open a context manager (auto-closes when done)
    with st.expander("📖 What is a Violin Plot?"):  # Open context manager
        # Display text or data in the Streamlit UI
        st.write("""  # Display text or data in the Streamlit UI
        # Execute this operation
        A Violin plot combines a boxplot and a KDE. The width of the violin at   # Execute this statement
        # Execute this operation
        each value shows the density (how common that value is). The inner box   # Execute this statement
        # Execute this operation
        shows the median and interquartile range.  # Execute this statement
        """)


# Define the _plot_bar function
def _plot_bar(df: pd.DataFrame, cat_col: str, num_col: str):  # Define the _plot_bar function
    """Create and display a bar chart of a numeric column grouped by a category."""
    # Group by the categorical column and compute the mean of the numeric column
    # Group DataFrame rows by a column for aggregate operations
    grouped = df.groupby(cat_col)[num_col].mean().reset_index()  # Group data by a column for aggregation

    # Create an interactive bar chart with plotly
    # Draw a bar chart
    fig = px.bar(  # Store result in fig
        grouped,               # Data source — grouped means
        x=cat_col,             # X-axis: category names
        y=num_col,             # Y-axis: mean values
        title=f"Mean {num_col} by {cat_col}",  # Chart title
        template="plotly_white",               # Clean theme
        color=cat_col,                         # Different color per category
        text_auto=".2f",                       # Show values on top of bars
    )
    st.plotly_chart(fig, use_container_width=True)  # Render in Streamlit
