import streamlit as st
import pandas as pd
import os
import pickle
from io import BytesIO

from pycaret.classification import (
    setup, compare_models, pull,
    finalize_model, plot_model, save_model
)


st.set_page_config(
    page_title="Clarity ML",
    layout="wide",
    page_icon="📊"
)


with st.sidebar:
    st.title("📊 Clarity ML")

    choice = st.radio(
        "Navigation",
        ["Upload", "EDA", "ML", "Download"]
    )

    st.info("Upload → Explore → Train Model → Download")


DATA_PATH = "data.csv"

df = None
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)


if choice == "Upload":
    st.title("📂 Upload Dataset")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df.to_csv(DATA_PATH, index=False)

        st.success("Dataset uploaded successfully!")
        st.dataframe(df.head())


elif choice == "EDA":
    st.title("📊 Exploratory Data Analysis")

    if df is None:
        st.warning("Please upload a dataset first.")
    else:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Basic Statistics")
        st.dataframe(df.describe())

        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum())

        st.subheader("Column Info")
        st.write(df.dtypes)

elif choice == "ML":
    st.title("🤖 AutoML (PyCaret)")

    if df is None:
        st.warning("Please upload a dataset first.")
    else:
        target = st.selectbox("Select Target Column", df.columns)

        if st.button("Train Model"):

            st.info("Initializing PyCaret...")

            setup(
                data=df,
                target=target,
                session_id=123,
                silent=True,
                verbose=False
            )

            st.success("Setup completed")

            st.subheader("Comparing Models...")
            best_model = compare_models()

            st.write("### Best Model")
            st.write(best_model)

            final_model = finalize_model(best_model)
            st.session_state["model"] = final_model

            st.subheader("Model Metrics")
            st.dataframe(pull())

            
            st.subheader("Confusion Matrix")
            try:
                plot_model(final_model, plot="confusion_matrix", display_format="streamlit")
            except:
                st.warning("Plot not available for this model")

            st.subheader("AUC Curve")
            try:
                plot_model(final_model, plot="auc", display_format="streamlit")
            except:
                st.warning("AUC not available")

            save_model(final_model, "best_model")
            st.success("Model saved successfully!")


elif choice == "Download":
    st.title("⬇️ Download Model")

    if "model" not in st.session_state:
        st.warning("Train a model first.")
    else:
        buffer = BytesIO()
        pickle.dump(st.session_state["model"], buffer)
        buffer.seek(0)

        st.download_button(
            label="Download Model (.pkl)",
            data=buffer,
            file_name="best_model.pkl",
            mime="application/octet-stream"
        )
