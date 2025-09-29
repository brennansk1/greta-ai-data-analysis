import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from greta_core.narratives.summary_narratives import generate_hypothesis_narrative, generate_summary_narrative

def show():
    st.title("📊 Analysis Results")
    st.markdown("---")

    if st.session_state.hypotheses is None:
        st.error("❌ No analysis results available. Please run the analysis first.")
        if st.button("Go to Analysis"):
            st.session_state.page = 'analysis'
            st.rerun()
        return

    hypotheses = st.session_state.hypotheses
    feature_names = st.session_state.feature_names
    target_col = st.session_state.target_column
    df = st.session_state.cleaned_data

    st.header("🎯 Top Insights")

    # Summary narrative
    with st.expander("📝 Analysis Summary", expanded=True):
        summary = generate_summary_narrative(hypotheses, feature_names)
        st.write(summary)

    # Results overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Hypotheses", len(hypotheses))
    with col2:
        avg_significance = sum(h['significance'] for h in hypotheses) / len(hypotheses)
        st.metric("Avg Significance", f"{avg_significance:.2f}")
    with col3:
        top_fitness = hypotheses[0]['fitness'] if hypotheses else 0
        st.metric("Top Fitness Score", f"{top_fitness:.2f}")

    # Hypotheses ranking
    st.header("🏆 Ranked Hypotheses")

    for i, hyp in enumerate(hypotheses[:5], 1):  # Show top 5
        with st.expander(f"#{i} Hypothesis (Fitness: {hyp['fitness']:.3f})", expanded=(i==1)):

            # Narrative
            narrative = generate_hypothesis_narrative(hyp, feature_names)
            st.write("**Insight:**")
            st.info(narrative)

            # Details
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Significance", f"{hyp['significance']:.3f}")
            with col2:
                st.metric("Effect Size", f"{hyp['effect_size']:.3f}")
            with col3:
                st.metric("Coverage", f"{hyp['coverage']:.3f}")
            with col4:
                st.metric("Features Used", len(hyp['features']))

            # Feature list
            selected_features = [feature_names[j] for j in hyp['features']]
            st.write("**Features in this hypothesis:**")
            st.write(", ".join(selected_features))

            # Visualization
            if len(selected_features) >= 1:
                st.subheader("📈 Feature Relationships")

                # Scatter plot for single feature
                if len(selected_features) == 1:
                    feature = selected_features[0]
                    fig = px.scatter(
                        df,
                        x=feature,
                        y=target_col,
                        title=f"Relationship: {feature} vs {target_col}",
                        trendline="ols"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Correlation heatmap for multiple features
                elif len(selected_features) > 1:
                    corr_data = df[selected_features + [target_col]].corr()
                    fig = px.imshow(
                        corr_data,
                        title="Feature Correlations",
                        text_auto=True,
                        aspect="auto"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Pair plot
                    fig_pair = px.scatter_matrix(
                        df[selected_features + [target_col]],
                        title="Feature Pair Relationships"
                    )
                    st.plotly_chart(fig_pair, use_container_width=True)

            # Detailed Analysis Expander
            with st.expander("🔍 Detailed Analysis", expanded=False):
                st.markdown("**Exact Statistical Metrics:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    p_value = 1 - hyp['significance']
                    st.metric("P-Value", f"{p_value:.4f}")
                with col2:
                    st.metric("Effect Size", f"{hyp['effect_size']:.4f}")
                with col3:
                    st.metric("Coverage (R²)", f"{hyp['coverage']:.4f}")

                st.markdown(f"**Analysis Type:** {hyp['analysis_type'].replace('_', ' ').title()}")

                st.markdown("**Data Segment Details:**")
                if len(selected_features) > 0:
                    segment_df = df[selected_features + [target_col]].describe()
                    st.dataframe(segment_df, use_container_width=True)

                    # Show correlations with target
                    correlations = df[selected_features + [target_col]].corr()[target_col].drop(target_col)
                    st.write("**Feature Correlations with Target:**")
                    for feat, corr in correlations.items():
                        st.write(f"- {feat}: {corr:.4f}")
                else:
                    st.write("No features selected for this hypothesis.")

    # Fitness distribution
    st.header("📊 Analysis Metrics")
    fitness_scores = [h['fitness'] for h in hypotheses]
    significance_scores = [h['significance'] for h in hypotheses]
    effect_sizes = [h['effect_size'] for h in hypotheses]

    col1, col2 = st.columns(2)

    with col1:
        fig_fitness = px.histogram(
            x=fitness_scores,
            title="Distribution of Hypothesis Fitness Scores",
            labels={'x': 'Fitness Score'}
        )
        st.plotly_chart(fig_fitness, use_container_width=True)

    with col2:
        # Scatter of significance vs effect size
        fig_scatter = px.scatter(
            x=significance_scores,
            y=effect_sizes,
            title="Significance vs Effect Size",
            labels={'x': 'Significance', 'y': 'Effect Size'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Export options
    st.header("💾 Export Results")
    if st.button("Download Results as CSV"):
        results_df = pd.DataFrame([{
            'rank': i+1,
            'features': ', '.join([feature_names[j] for j in h['features']]),
            'significance': h['significance'],
            'effect_size': h['effect_size'],
            'coverage': h['coverage'],
            'fitness': h['fitness'],
            'narrative': generate_hypothesis_narrative(h, feature_names)
        } for i, h in enumerate(hypotheses)])

        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="greta_analysis_results.csv",
            mime="text/csv"
        )

    # Start new analysis
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Start New Analysis", type="secondary", use_container_width=True):
            st.session_state.page = 'welcome'
            st.rerun()