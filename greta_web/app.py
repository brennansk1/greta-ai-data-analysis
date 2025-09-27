import sys
sys.path.append('..')

import streamlit as st
from greta_web.pages import welcome, data_upload, data_health, analysis, results, login, register, project_management
from greta_web.auth import is_logged_in, logout_user

# Set page configuration
st.set_page_config(
    page_title="Greta Web App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'login'  # Start with login
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    if 'hypotheses' not in st.session_state:
        st.session_state.hypotheses = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None

init_session_state()

# Sidebar navigation
st.sidebar.title("Greta Web App")
st.sidebar.markdown("---")

if is_logged_in():
    if st.sidebar.button("Logout"):
        logout_user()
        st.rerun()

    pages = {
        'Welcome & Project Hub': 'welcome',
        'Project Management': 'project_management',
        'Data Upload': 'data_upload',
        'Data Health Dashboard': 'data_health',
        'Analysis Dashboard': 'analysis',
        'Results Page': 'results'
    }

    selected_page = st.sidebar.selectbox(
        "Navigate to:",
        list(pages.keys()),
        index=list(pages.values()).index(st.session_state.page) if st.session_state.page in pages.values() else 0
    )

    st.session_state.page = pages[selected_page]
else:
    auth_pages = {
        'Login': 'login',
        'Register': 'register'
    }

    selected_page = st.sidebar.selectbox(
        "Navigate to:",
        list(auth_pages.keys()),
        index=list(auth_pages.values()).index(st.session_state.page) if st.session_state.page in auth_pages.values() else 0
    )

    st.session_state.page = auth_pages[selected_page]

# Progress indicator
progress_steps = ['welcome', 'data_upload', 'data_health', 'analysis', 'results']
current_step = progress_steps.index(st.session_state.page) if st.session_state.page in progress_steps else 0
progress = (current_step + 1) / len(progress_steps)
st.sidebar.progress(progress)
st.sidebar.markdown(f"**Step {current_step + 1} of {len(progress_steps)}**")

# Render the selected page
if st.session_state.page == 'login':
    login.show()
elif st.session_state.page == 'register':
    register.show()
else:
    # Require login for other pages
    if not is_logged_in():
        st.session_state.page = 'login'
        st.rerun()
    elif st.session_state.page == 'welcome':
        welcome.show()
    elif st.session_state.page == 'project_management':
        project_management.show()
    elif st.session_state.page == 'data_upload':
        data_upload.show()
    elif st.session_state.page == 'data_health':
        data_health.show()
    elif st.session_state.page == 'analysis':
        analysis.show()
    elif st.session_state.page == 'results':
        results.show()