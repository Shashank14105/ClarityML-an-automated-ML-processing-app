import streamlit as st
import pandas as pd
import os
import pickle
from io import BytesIO

from pycaret.classification import (
    setup, compare_models, pull,
    finalize_model, plot_model, save_model
)

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Clarity ML", layout="wide")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("Clarity ML")

    choice = st.radio(
        "Navigation",
        ["Upload", "Profiling", "ML", "Download", "HTML Report"]
    )

    st.info(
        "Upload dataset → Profile data → Train ML model → Download model/report"
    )


# ---------------- LOAD DATA ----------------
DATA_PATH = "sourcedata.csv"

df = None
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)


# ---------------- UPLOAD ----------------
if choice == "Upload":
    st.title("Upload Dataset")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df.to_csv(DATA_PATH, index=False)
        st.success("Dataset uploaded successfully!")
        st.dataframe(df)


# ---------------- PROFILING ----------------
elif choice == "Profiling":
    st.title("EDA Report")

    if df is None:
        st.warning("Upload dataset first.")
    else:
        profile = ProfileReport(df, explorative=True)
        st_profile_report(profile)


# ---------------- ML ----------------
elif choice == "ML":
    st.title("AutoML (PyCaret)")

    if df is None:
        st.warning("Upload dataset first.")

    else:
        target = st.selectbox("Select Target Column", df.columns)

        if st.button("Run ML"):

            st.info("Setting up PyCaret...")
            setup(data=df, target=target, session_id=123, silent=True)

            st.subheader("Model Comparison")

            best_model = compare_models()

            st.write("Best Model:", best_model)

            final_model = finalize_model(best_model)
            st.session_state["model"] = final_model

            st.success("Model trained successfully!")

            # Metrics table
            st.subheader("Model Metrics")
            st.dataframe(pull())

            # Plots (safe handling)
            st.subheader("Confusion Matrix")
            try:
                plot_model(final_model, plot="confusion_matrix", display_format="streamlit")
            except:
                st.warning("Confusion matrix not available.")

            st.subheader("ROC Curve")
            try:
                plot_model(final_model, plot="auc", display_format="streamlit")
            except:
                st.warning("ROC not available for this model.")

            # Save model
            save_model(final_model, "best_model")
            st.success("Model saved as best_model.pkl")


# ---------------- DOWNLOAD MODEL ----------------
elif choice == "Download":
    st.title("Download Model")

    if "model" not in st.session_state:
        st.warning("Train model first in ML section.")
    else:
        buffer = BytesIO()
        pickle.dump(st.session_state["model"], buffer)
        buffer.seek(0)

        st.download_button(
            "Download Model (.pkl)",
            buffer,
            file_name="best_model.pkl",
            mime="application/octet-stream"
        )


# ---------------- HTML REPORT ----------------
elif choice == "HTML Report":
    st.title("Download Report")

    if "model" not in st.session_state:
        st.warning("Train model first in ML section.")
    else:

        report_html = f"""
        <html>
        <head><title>Clarity ML Report</title></head>
        <body>
            <h1>AutoML Report</h1>

            <h2>Best Model</h2>
            <p>{st.session_state["model"]}</p>

            <h2>Metrics</h2>
            {pull().to_html()}
        </body>
        </html>
        """

        st.download_button(
            "Download HTML Report",
            report_html,
            file_name="ml_report.html",
            mime="text/html"
        )
