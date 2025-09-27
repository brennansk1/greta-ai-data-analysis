import streamlit as st
from greta_web.auth import register_user

def show():
    st.title("📝 Register for Greta Web App")
    st.markdown("---")

    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register")

        if submitted:
            if password != confirm_password:
                st.error("Passwords do not match")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters")
            elif not username or not email:
                st.error("Username and email are required")
            else:
                if register_user(username, email, password):
                    st.success("Registration successful! You are now logged in.")
                    st.session_state.page = 'welcome'
                    st.rerun()
                else:
                    st.error("Username or email already exists")

    st.markdown("---")
    if st.button("Already have an account? Login here"):
        st.session_state.page = 'login'
        st.rerun()