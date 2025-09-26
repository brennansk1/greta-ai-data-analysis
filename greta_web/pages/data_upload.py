import streamlit as st
import pandas as pd
import tempfile
import os
from sqlalchemy import create_engine, text
from greta_core.ingestion import load_csv, load_excel, detect_schema, validate_data

def show():
    st.title("📤 Data Upload")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📁 File Upload", "🗄️ Database Connection"])

    with tab1:
        st.header("Upload Your Dataset")
        st.markdown("Greta supports CSV and Excel files. Drag and drop or browse to select your file.")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help="Supported formats: CSV, Excel (.xlsx, .xls)"
        )

        if uploaded_file is not None:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Load data based on file type
                with st.spinner("Loading data..."):
                    if uploaded_file.name.endswith('.csv'):
                        df = load_csv(tmp_path)
                    else:
                        df = load_excel(tmp_path)

                # Store raw data
                st.session_state.raw_data = df
                st.session_state.feature_names = list(df.columns)

                st.success(f"✅ Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns")

                # Data preview
                st.subheader("📊 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

                # Schema detection
                st.subheader("🔍 Data Schema")
                schema = detect_schema(df)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", schema['shape'][0])
                with col2:
                    st.metric("Columns", schema['shape'][1])
                with col3:
                    st.metric("Data Types", len(set(schema['dtypes'].values())))

                # Column details
                st.markdown("**Column Details:**")
                for col, info in schema['columns'].items():
                    with st.expander(f"📋 {col}"):
                        st.write(f"**Type:** {info['dtype']}")
                        st.write(f"**Null Count:** {info['null_count']}")
                        st.write(f"**Unique Values:** {info['unique_count']}")
                        if info['sample_values']:
                            st.write(f"**Sample Values:** {info['sample_values']}")

                # Validation
                st.subheader("⚠️ Validation Results")
                warnings = validate_data(df)

                if warnings:
                    for warning in warnings:
                        st.warning(warning)
                else:
                    st.success("✅ No validation issues found!")

                # Proceed button
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🔍 Review Data Health", type="primary", use_container_width=True):
                        st.session_state.page = 'data_health'
                        st.rerun()

            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.session_state.raw_data = None

            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        else:
            st.info("👆 Please upload a CSV or Excel file to continue")
            st.session_state.raw_data = None

            # Sample data option (for demo)
            st.markdown("---")
            st.subheader("🎯 Try Sample Data")
            if st.button("Load Sample Dataset"):
                # Create sample data
                import numpy as np
                np.random.seed(42)
                sample_data = {
                    'age': np.random.normal(35, 10, 100),
                    'income': np.random.normal(50000, 15000, 100),
                    'education_years': np.random.normal(16, 2, 100),
                    'satisfaction': np.random.normal(7, 1.5, 100)
                }
                df = pd.DataFrame(sample_data)
                st.session_state.raw_data = df
                st.session_state.feature_names = list(df.columns)
                st.success("✅ Sample data loaded!")
                st.dataframe(df.head(10), use_container_width=True)
                if st.button("Continue with Sample Data"):
                    st.session_state.page = 'data_health'
                    st.rerun()

    with tab2:
        st.header("Connect to Database")
        st.markdown("Connect to PostgreSQL, MySQL, or SQLite databases to load your data.")

        # Database type selection
        db_type = st.selectbox(
            "Database Type",
            ["PostgreSQL", "MySQL", "SQLite"],
            help="Select the type of database you want to connect to"
        )

        # Connection form based on db_type
        with st.form("db_connection_form"):
            if db_type == "SQLite":
                db_file = st.text_input("Database File Path", placeholder="e.g., /path/to/database.db")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    host = st.text_input("Host", value="localhost")
                    port = st.number_input("Port", value=5432 if db_type == "PostgreSQL" else 3306, min_value=1, max_value=65535)
                with col2:
                    database = st.text_input("Database Name")
                    username = st.text_input("Username")
                password = st.text_input("Password", type="password")

            # Connect button
            submitted = st.form_submit_button("🔗 Connect to Database")

        if submitted:
            try:
                if db_type == "SQLite":
                    if not db_file:
                        st.error("Please provide a database file path")
                        st.stop()
                    engine_url = f"sqlite:///{db_file}"
                elif db_type == "PostgreSQL":
                    if not all([host, database, username]):
                        st.error("Please fill in all required fields")
                        st.stop()
                    engine_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
                elif db_type == "MySQL":
                    if not all([host, database, username]):
                        st.error("Please fill in all required fields")
                        st.stop()
                    engine_url = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"

                # Create engine
                engine = create_engine(engine_url)
                st.session_state.db_engine = engine
                st.session_state.db_type = db_type

                # Get table list
                with engine.connect() as conn:
                    if db_type == "SQLite":
                        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
                        tables = [row[0] for row in result.fetchall()]
                    elif db_type == "PostgreSQL":
                        result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"))
                        tables = [row[0] for row in result.fetchall()]
                    elif db_type == "MySQL":
                        result = conn.execute(text("SHOW TABLES;"))
                        tables = [row[0] for row in result.fetchall()]

                st.session_state.db_tables = tables
                st.success(f"✅ Connected to {db_type} database successfully!")
                st.info(f"Found {len(tables)} tables: {', '.join(tables)}")

            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
                st.session_state.db_engine = None

        # If connected, show table selection
        if 'db_engine' in st.session_state and st.session_state.db_engine is not None:
            st.subheader("Select Table")
            selected_table = st.selectbox("Choose a table", st.session_state.db_tables)

            if selected_table:
                # Query preview
                preview_query = f"SELECT * FROM {selected_table} LIMIT 10;"
                st.subheader("Query Preview")
                st.code(preview_query, language="sql")

                if st.button("🔍 Preview Data"):
                    try:
                        with st.session_state.db_engine.connect() as conn:
                            result = conn.execute(text(preview_query))
                            columns = result.keys()
                            data = result.fetchall()
                            df_preview = pd.DataFrame(data, columns=columns)
                            st.dataframe(df_preview, use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Error previewing data: {str(e)}")

                # Load full table
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("📥 Load Full Table", type="primary", use_container_width=True):
                        try:
                            with st.spinner("Loading data..."):
                                query = f"SELECT * FROM {selected_table};"
                                df = pd.read_sql(query, st.session_state.db_engine)
                                st.session_state.raw_data = df
                                st.session_state.feature_names = list(df.columns)
                                st.success(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns from {selected_table}")
                                st.dataframe(df.head(10), use_container_width=True)
                                if st.button("Continue to Data Health Review"):
                                    st.session_state.page = 'data_health'
                                    st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error loading table: {str(e)}")