import streamlit as st
from greta_web.database import get_user_projects
from greta_web.auth import get_current_user

def show():
    user = get_current_user()
    st.title(f"🔍 Welcome back, {user['username']}!")
    st.markdown("---")

    st.header("Your Projects Dashboard")

    projects = get_user_projects(user['id'])

    if projects:
        owned_projects = [p for p in projects if p['permission'] == 'admin']
        shared_projects = [p for p in projects if p['permission'] != 'admin']

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📁 Your Projects")
            if owned_projects:
                for project in owned_projects:
                    with st.expander(f"{project['name']} ({project['permission']})"):
                        st.write(f"Description: {project['description'] or 'No description'}")
                        st.write(f"Created: {project['created_at']}")
                        if st.button(f"Open {project['name']}", key=f"open_{project['id']}"):
                            # For now, just go to data upload
                            st.session_state.page = 'data_upload'
                            st.rerun()
            else:
                st.info("You haven't created any projects yet.")

        with col2:
            st.subheader("🤝 Shared Projects")
            if shared_projects:
                for project in shared_projects:
                    with st.expander(f"{project['name']} ({project['permission']})"):
                        st.write(f"Description: {project['description'] or 'No description'}")
                        st.write(f"Created: {project['created_at']}")
                        if st.button(f"Open {project['name']}", key=f"open_shared_{project['id']}"):
                            st.session_state.page = 'data_upload'
                            st.rerun()
            else:
                st.info("No projects shared with you yet.")
    else:
        st.info("You don't have any projects yet. Create your first project below!")

    st.markdown("---")
    st.subheader("Create New Project")
    with st.form("create_project"):
        project_name = st.text_input("Project Name")
        project_desc = st.text_area("Description (optional)")
        submitted = st.form_submit_button("Create Project")

        if submitted and project_name:
            from greta_web.database import create_project
            project_id = create_project(project_name, project_desc, user['id'])
            if project_id:
                st.success("Project created successfully!")
                st.rerun()
            else:
                st.error("Failed to create project")

    st.markdown("---")
    st.caption("Powered by Greta Core Engine | Built with Streamlit")