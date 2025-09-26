import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from greta_core.hypothesis_search import generate_hypotheses

def show():
    st.title("🎯 Analysis Dashboard")
    st.markdown("---")

    if st.session_state.cleaned_data is None:
        st.error("❌ No cleaned data available. Please complete data upload and health check.")
        if st.button("Go to Data Upload"):
            st.session_state.page = 'data_upload'
            st.rerun()
        return

    df = st.session_state.cleaned_data

    st.header("Select Target Variable")
    st.markdown("Choose the column you want to understand or predict. This should be a numeric variable.")

    # Target variable selection
    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()

    if not numeric_cols:
        st.error("❌ No numeric columns found. Please check your data.")
        return

    target_col = st.selectbox(
        "Target Variable",
        numeric_cols,
        help="Select the variable you want to analyze relationships for"
    )

    st.session_state.target_column = target_col

    # Show target distribution
    st.subheader(f"📊 Distribution of {target_col}")
    fig = px.histogram(df, x=target_col, title=f"Distribution of {target_col}", marginal="box")
    st.plotly_chart(fig, use_container_width=True)

    # Analysis parameters
    st.header("⚙️ Analysis Parameters")
    st.markdown("Configure the genetic algorithm parameters for hypothesis search.")

    col1, col2, col3 = st.columns(3)

    with col1:
        pop_size = st.slider(
            "Population Size",
            min_value=50,
            max_value=200,
            value=100,
            step=10,
            help="Number of candidate solutions in each generation"
        )

    with col2:
        num_generations = st.slider(
            "Number of Generations",
            min_value=10,
            max_value=100,
            value=50,
            step=5,
            help="How many iterations the algorithm will run"
        )

    with col3:
        cx_prob = st.slider(
            "Crossover Probability",
            min_value=0.5,
            max_value=0.9,
            value=0.7,
            step=0.1,
            help="Probability of combining parent solutions"
        )

    st.markdown("---")

    # Feature selection preview
    feature_cols = [col for col in df.columns if col != target_col]
    st.subheader("📋 Analysis Summary")
    st.write(f"**Target Variable:** {target_col}")
    st.write(f"**Number of Features:** {len(feature_cols)}")
    st.write(f"**Total Samples:** {len(df)}")

    # Run analysis
    if st.button("🚀 Find Insights", type="primary", use_container_width=True):
        with st.spinner("Running genetic algorithm analysis... This may take a few minutes."):

            # Prepare data
            X = df[feature_cols].values
            y = df[target_col].values

            # Handle any remaining NaN values
            if np.isnan(X).any() or np.isnan(y).any():
                st.warning("⚠️ Data contains NaN values. Filling with column means.")
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='mean')
                X = imputer.fit_transform(X)
                y = pd.Series(y).fillna(pd.Series(y).mean()).values

            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Run analysis (this might take time)
            try:
                hypotheses = generate_hypotheses(
                    X, y,
                    pop_size=pop_size,
                    num_generations=num_generations,
                    cx_prob=cx_prob,
                    mut_prob=0.2
                )

                # Store results
                st.session_state.hypotheses = hypotheses
                st.session_state.feature_names = feature_cols

                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")

                st.success(f"🎉 Found {len(hypotheses)} potential hypotheses!")
                st.balloons()

                # Auto-proceed to results
                st.session_state.page = 'results'
                st.rerun()

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                progress_bar.empty()
                status_text.empty()

    # Preview button
    if st.button("👀 Preview Data for Analysis"):
        st.subheader("Data Preview")
        preview_df = df[feature_cols + [target_col]].head(10)
        st.dataframe(preview_df, use_container_width=True)

        st.subheader("Correlation with Target")
        correlations = df[feature_cols + [target_col]].corr()[target_col].drop(target_col).sort_values(ascending=False)
        st.bar_chart(correlations)