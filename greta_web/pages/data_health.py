import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from greta_core.preprocessing.profiling_stats import profile_data
from greta_core.preprocessing.data_normalization import normalize_data_types
from greta_core.preprocessing.missing_value_handling import handle_missing_values
from greta_core.preprocessing.outlier_detection import detect_outliers, remove_outliers

def show():
    st.title("🏥 Data Health Dashboard")
    st.markdown("---")

    if st.session_state.raw_data is None:
        st.error("❌ No data uploaded. Please go back to Data Upload.")
        if st.button("Go to Data Upload"):
            st.session_state.page = 'data_upload'
            st.rerun()
        return

    df = st.session_state.raw_data

    # Initialize cleaned data
    if st.session_state.cleaned_data is None:
        st.session_state.cleaned_data = df.copy()

    st.header("📊 Data Quality Overview")

    # Profile data
    with st.spinner("Analyzing data quality..."):
        profile = profile_data(df)

    # Health metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_missing = sum(col['null_count'] for col in profile['columns'].values())
        st.metric("Total Missing Values", total_missing)
    with col2:
        numeric_cols = [col for col, info in profile['columns'].items() if pd.api.types.is_numeric_dtype(pd.Series(dtype=info['dtype']))]
        st.metric("Numeric Columns", len(numeric_cols))
    with col3:
        categorical_cols = [col for col, info in profile['columns'].items() if not pd.api.types.is_numeric_dtype(pd.Series(dtype=info['dtype']))]
        st.metric("Categorical Columns", len(categorical_cols))
    with col4:
        outlier_cols = [col for col in df.select_dtypes(include=[float, int]).columns if detect_outliers(df, method='iqr')[col]]
        st.metric("Columns with Outliers", len(outlier_cols))

    # Missing data visualization
    st.subheader("🔍 Missing Data Analysis")
    missing_data = pd.DataFrame({
        'Column': list(profile['columns'].keys()),
        'Missing Count': [col['null_count'] for col in profile['columns'].values()],
        'Missing Percentage': [col['null_percentage'] for col in profile['columns'].values()]
    })

    fig_missing = px.bar(
        missing_data,
        x='Column',
        y='Missing Percentage',
        title="Missing Data by Column",
        labels={'Missing Percentage': 'Missing %'}
    )
    st.plotly_chart(fig_missing, use_container_width=True)

    # Data type distribution
    st.subheader("📋 Data Types")
    dtype_counts = pd.Series([info['dtype'] for info in profile['columns'].values()]).value_counts()
    fig_dtypes = px.pie(
        values=dtype_counts.values,
        names=dtype_counts.index,
        title="Data Type Distribution"
    )
    st.plotly_chart(fig_dtypes, use_container_width=True)

    # Distribution plots for numeric columns
    st.subheader("📈 Data Distributions")
    numeric_cols = df.select_dtypes(include=[float, int]).columns[:3]  # Show first 3

    if len(numeric_cols) > 0:
        for col in numeric_cols:
            fig = px.histogram(
                df,
                x=col,
                title=f"Distribution of {col}",
                marginal="box"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Cleaning options
    st.header("🧹 Data Cleaning Options")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Handle Missing Values")
        missing_strategy = st.selectbox(
            "Strategy",
            ['mean', 'median', 'mode', 'drop'],
            help="Choose how to handle missing values"
        )

        if st.button("Apply Missing Value Handling"):
            with st.spinner("Processing..."):
                cleaned = handle_missing_values(st.session_state.cleaned_data, strategy=missing_strategy)
                st.session_state.cleaned_data = cleaned
                st.success("✅ Missing values handled!")
                st.rerun()

    with col2:
        st.subheader("Handle Outliers")
        outlier_method = st.selectbox(
            "Detection Method",
            ['iqr', 'zscore'],
            help="Choose outlier detection method"
        )

        if st.button("Remove Outliers"):
            with st.spinner("Processing..."):
                outliers = detect_outliers(st.session_state.cleaned_data, method=outlier_method)
                cleaned = remove_outliers(st.session_state.cleaned_data, outliers)
                st.session_state.cleaned_data = cleaned
                st.success(f"✅ Removed outliers from {len(outliers)} columns!")
                st.rerun()

    # Normalize data types
    if st.button("Normalize Data Types"):
        with st.spinner("Processing..."):
            cleaned = normalize_data_types(st.session_state.cleaned_data)
            st.session_state.cleaned_data = cleaned
            st.success("✅ Data types normalized!")
            st.rerun()

    # Reset to original
    if st.button("Reset to Original Data"):
        st.session_state.cleaned_data = df.copy()
        st.success("✅ Reset to original data!")
        st.rerun()

    # Current data preview
    st.header("📋 Current Data Preview")
    st.dataframe(st.session_state.cleaned_data.head(10), use_container_width=True)

    # Proceed to analysis
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 Proceed to Analysis", type="primary", use_container_width=True):
            st.session_state.page = 'analysis'
            st.rerun()