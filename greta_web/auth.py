import streamlit as st
from .database import create_user, verify_password, get_user_by_id

def login_user(username, password):
    user = verify_password(username, password)
    if user:
        st.session_state.user_id = user['id']
        st.session_state.username = user['username']
        return True
    return False

def register_user(username, email, password):
    user_id = create_user(username, email, password)
    if user_id:
        st.session_state.user_id = user_id
        st.session_state.username = username
        return True
    return False

def logout_user():
    if 'user_id' in st.session_state:
        del st.session_state.user_id
    if 'username' in st.session_state:
        del st.session_state.username
    # Reset other session state as needed
    st.session_state.page = 'login'

def is_logged_in():
    return 'user_id' in st.session_state

def get_current_user():
    if is_logged_in():
        return get_user_by_id(st.session_state.user_id)
    return None

def require_login():
    if not is_logged_in():
        st.session_state.page = 'login'
        st.rerun()