import streamlit as st
import pandas as pd
import os

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from pycaret.classification import setup, compare_models, pull, save_model, plot_model, finalize_model

#Download
import pickle
from io import BytesIO

#ML Report


# PAGE CONFIG
st.set_page_config(page_title="Clarity ML", layout="wide")

# SIDEBAR
with st.sidebar:
    st.image("assets/banner1.png", use_container_width=True)
    st.title("Clarity ML")

    choice = st.radio(
        "Navigation",
        ["Upload", "Profiling", "ML", "Download","HTML Report"]
    )

    st.info(
        "Clarity ML helps you upload datasets, explore data, generate automated profiling reports, "
        "train ML models using PyCaret, and download outputs easily."
    )

# LOAD DATA
df = None

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv")


# UPLOAD
if choice == "Upload":
    st.title("Upload Your Dataset")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df.to_csv("sourcedata.csv", index=False)
        st.success("Dataset uploaded successfully!")
        st.dataframe(df)


# PROFILING
elif choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")

    if df is None:
        st.warning("Please upload a dataset first.")
    else:
        profile = ProfileReport(df, explorative=True)
        st_profile_report(profile)


# ML
elif choice == "ML":
    st.title("Machine Learning with PyCaret")

    if df is None:
        st.warning("Please upload a dataset first.")
    else:
        target = st.selectbox("Select Target Column", df.columns)

        if st.button("Run ML"):
            # Setup
            setup(data=df, target=target, session_id=123)

            setup_df = pull()
            st.subheader("Setup Summary")

            # Train models
            best_model = compare_models(include=["lr", "rf", "et", "dt", "knn"])

            final_model = finalize_model(best_model)

            # STORE MODEL 
            st.session_state["model"] = final_model

            st.subheader("📉 Confusion Matrix")
            plot_model(final_model, plot="confusion_matrix", display_format="streamlit")

            try:
                st.subheader("📊 ROC Curve")
                plot_model(final_model, plot="auc", display_format="streamlit")
            except Exception:
                st.warning("ROC/AUC not Available for this Model")

            compare_df = pull()
            st.subheader("Model Comparison")
            st.dataframe(compare_df)

            st.subheader("Best Model")
            st.write(final_model)

            # METRICS
            st.subheader("📈 Final Model Metrics")

            results = pull()
            st.dataframe(results)

            save_model(final_model, "best_model")
            st.success("Your Model is saved as best_model.pkl and can now be downloaded")


# DOWNLOAD
elif choice == "Download":
    st.subheader("📥 Download Trained Model")

    if df is None:
        st.warning("Please upload a dataset first.")

    elif "model" not in st.session_state:
        st.warning("Please train the model first in ML section.")

    else:
        model_buffer = BytesIO()
        pickle.dump(st.session_state["model"], model_buffer)
        model_buffer.seek(0)

        st.download_button(
            label="Download Model (.pkl)",
            data=model_buffer,
            file_name="best_model.pkl",
            mime="application/octet-stream"
        )


# HTML REPORT
elif choice == "HTML Report":
    st.subheader("📄 Download ML Report (HTML)")

    if "model" not in st.session_state:
        st.warning("Please train the model first in ML section.")

    else:
        report_html = f"""
        <html>
        <head>
            <title>ML Report</title>
        </head>
        <body>
            <h1>AutoML Report - Clarity ML</h1>

            <h2>Best Model</h2>
            <p>{str(st.session_state["model"])}</p>

            <h2>Metrics</h2>
            {pull().to_html()}

        </body>
        </html>
        """

        st.download_button(
            label="Download HTML Report",
            data=report_html,
            file_name="ml_report.html",
            mime="text/html"
        )