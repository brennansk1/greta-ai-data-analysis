import streamlit as st

def show():
    st.title("🔍 Welcome to Greta Web App")
    st.markdown("---")

    st.header("Discover Insights in Your Data with AI-Powered Analysis")

    st.markdown("""
    Greta is an intelligent data analysis tool that uses genetic algorithms to automatically explore your data
    and uncover meaningful relationships and patterns. No advanced statistical knowledge required!

    **What Greta can do for you:**
    - 🔄 **Automated Analysis**: Upload your data and let Greta find the most promising hypotheses
    - 📊 **Data Health Check**: Get instant feedback on data quality and cleaning suggestions
    - 🎯 **Targeted Insights**: Focus analysis on specific outcomes you're interested in
    - 📈 **Interactive Visualizations**: Explore results with beautiful, interactive charts
    - 📝 **Plain-English Explanations**: Understand findings in simple, actionable language

    **How it works:**
    1. **Upload** your CSV or Excel data
    2. **Review** data quality and apply cleaning if needed
    3. **Select** your target variable and analysis parameters
    4. **Discover** ranked insights and narratives
    """)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("🚀 Start New Analysis", type="primary", use_container_width=True):
            # Reset session state for new project
            st.session_state.raw_data = None
            st.session_state.cleaned_data = None
            st.session_state.target_column = None
            st.session_state.hypotheses = None
            st.session_state.results = None
            st.session_state.feature_names = None
            st.session_state.page = 'data_upload'
            st.rerun()

    st.markdown("---")
    st.subheader("Quick Tips")
    st.info("""
    💡 **Data Format**: Greta works best with tabular data (CSV, Excel) with clear column headers
    💡 **Data Size**: Start with datasets up to 10,000 rows for optimal performance
    💡 **Target Variable**: Choose a numeric column you want to understand or predict
    💡 **Privacy**: Your data stays local - no uploads to external servers
    """)

    st.markdown("---")
    st.caption("Powered by Greta Core Engine | Built with Streamlit")