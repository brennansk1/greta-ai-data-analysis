import streamlit as st
from greta_web.auth import login_user

def show():
    st.title("🔐 Login to Greta Web App")
    st.markdown("---")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if login_user(username, password):
                st.success("Login successful!")
                st.session_state.page = 'welcome'
                st.rerun()
            else:
                st.error("Invalid username or password")

    st.markdown("---")
    if st.button("Don't have an account? Register here"):
        st.session_state.page = 'register'
        st.rerun()