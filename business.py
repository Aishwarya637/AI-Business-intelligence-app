import io
import os
import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# ---------------------------------------------------------
# Environment & Page Configuration
# ---------------------------------------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="AI Business Intelligence Copilot",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI Business Intelligence Copilot")
st.markdown(
    "Upload your dataset (CSV/Excel), ask questions in natural language, and get instant SQL/Pandas analysis, charts, and executive summaries."
)

# Check if API key is present in environment
if not api_key:
    st.error(
        "⚠️ `GOOGLE_API_KEY` not found in your environment or `.env` file. Please configure it to use natural language queries."
    )

# ---------------------------------------------------------
# Sidebar: File Upload
# ---------------------------------------------------------
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file", type=["csv", "xlsx"]
)


@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)


if uploaded_file is not None:
    df = load_data(uploaded_file)

    # Register DataFrame with SQLite in-memory database for SQL queries
    conn = sqlite3.connect(":memory:")
    df.to_sql("dataset", conn, index=False, if_exists="replace")

    st.success("File successfully loaded!")

    # ---------------------------------------------------------
    # Dataset Preview & KPI Dashboard
    # ---------------------------------------------------------
    with st.expander("🔍 Preview Raw Data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    st.subheader("📈 Automated KPI Dashboard")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if numeric_cols:
        cols = st.columns(min(len(numeric_cols), 4))
        for i, col in enumerate(numeric_cols[:4]):
            total_val = df[col].sum()
            cols[i].metric(label=f"Total {col}", value=f"{total_val:,.2f}")

    # ---------------------------------------------------------
    # Natural Language Query Section
    # ---------------------------------------------------------
    st.subheader("💬 Ask Your Data")
    user_query = st.text_input(
        "Ask a business question (e.g., 'Which product has the highest profit margin?' or 'Show monthly sales trends')"
    )

    if user_query:
        if not api_key:
            st.error(
                "Please add your Google API key to the `.env` file to run queries."
            )
        else:
            # Configure Google GenAI
            genai.configure(api_key=api_key)
            # Using gemini-1.5-flash or gemini-pro
            model = genai.GenerativeModel("gemini-flash-latest")

            # Generate schema info for the LLM
            schema_info = f"Columns and types:\n{df.dtypes.to_string()}\n\nSample data:\n{df.head(3).to_string()}"

            prompt = f"""
            You are an expert Data Analyst and SQL/Pandas developer. 
            The user is asking a question about a dataset named 'dataset' in SQLite.
            
            {schema_info}
            
            User Question: {user_query}
            
            Provide your response strictly in the following JSON format without any extra markdown:
            {{
                "analysis_type": "sql" or "pandas",
                "code": "executable python code using pandas (dataframe is named df) OR executable SQLite SQL query using table 'dataset'",
                "explanation": "Brief explanation of the insight or trend",
                "chart_suggestion": "bar", "line", "scatter", or "none"
            }}
            """

            with st.spinner("Analyzing data and generating insights..."):
                try:
                    response = model.generate_content(prompt)
                    response_text = response.text

                    import json

                    # Clean up code block ticks if returned by LLM
                    clean_json = (
                        response_text.replace("```json", "")
                        .replace("```", "")
                        .strip()
                    )
                    parsed = json.loads(clean_json)

                    st.markdown("### 📋 Executive Summary & Insights")
                    st.info(parsed["explanation"])

                    st.markdown("### 💻 Generated Code")
                    st.code(
                        parsed["code"],
                        language=(
                            "python"
                            if parsed["analysis_type"] == "pandas"
                            else "sql"
                        ),
                    )

                    # Execute code safely using sqlite3
                    if parsed["analysis_type"] == "sql":
                        result_df = pd.read_sql_query(parsed["code"], conn)
                    else:
                        local_vars = {"df": df, "pd": pd}
                        exec(f"result_df = {parsed['code']}", globals(), local_vars)
                        result_df = local_vars["result_df"]

                    st.markdown("### 📊 Query Results")
                    st.dataframe(result_df, use_container_width=True)

                    # Render chart if applicable
                    if (
                        parsed["chart_suggestion"] != "none"
                        and not result_df.empty
                    ):
                        st.markdown("### 📈 Visual Representation")
                        chart_type = parsed["chart_suggestion"]
                        cols = result_df.columns.tolist()
                        if len(cols) >= 2:
                            if chart_type == "bar":
                                fig = px.bar(
                                    result_df, x=cols[0], y=cols[1]
                                )
                            elif chart_type == "line":
                                fig = px.line(
                                    result_df, x=cols[0], y=cols[1]
                                )
                            else:
                                fig = px.scatter(
                                    result_df, x=cols[0], y=cols[1]
                                )
                            st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    st.text(
                        response_text if "response_text" in locals() else ""
                    )

else:
    st.info(
        "👈 Please upload a CSV or Excel file in the sidebar to get started."
    )