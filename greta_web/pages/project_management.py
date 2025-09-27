import streamlit as st
from greta_web.database import get_project_by_id, get_project_users, share_project, get_user_by_username
from greta_web.auth import get_current_user

def show():
    st.title("📁 Project Management")
    st.markdown("---")

    user = get_current_user()

    # Assume project_id is in session or query param, for now use a selectbox
    # In real app, pass project_id via session or URL

    # For demo, list user's projects
    from greta_web.database import get_user_projects
    projects = get_user_projects(user['id'])
    owned_projects = [p for p in projects if p['permission'] == 'admin']

    if not owned_projects:
        st.info("You don't have any projects to manage.")
        return

    project_names = [p['name'] for p in owned_projects]
    selected_project_name = st.selectbox("Select Project to Manage", project_names)
    selected_project = next(p for p in owned_projects if p['name'] == selected_project_name)

    st.header(f"Managing: {selected_project['name']}")
    st.write(f"Description: {selected_project['description'] or 'No description'}")

    # Share project
    st.subheader("Share Project")
    with st.form("share_project"):
        username = st.text_input("Username to share with")
        permission = st.selectbox("Permission", ["read", "write"])
        submitted = st.form_submit_button("Share")

        if submitted and username:
            user_to_share = get_user_by_username(username)
            if user_to_share:
                if share_project(selected_project['id'], user_to_share['id'], permission):
                    st.success(f"Project shared with {username}")
                else:
                    st.error("Failed to share project")
            else:
                st.error("User not found")

    # Current users
    st.subheader("Current Users")
    users = get_project_users(selected_project['id'])
    for u in users:
        st.write(f"- {u['username']}: {u['permission']}")