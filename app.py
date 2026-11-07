
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Steel Plate Fault Classification",
                   page_icon="⚙️", layout="wide")

st.title("Steel Plate Fault Classification")
st.write("Machine Learning Based Industrial Defect Analysis")

models = {
    "Logistic Regression": "Logistic_Regression.joblib",
    "Decision Tree": "Decision_Tree.joblib",
    "KNN": "KNN.joblib",
    "Naive Bayes": "Naive_Bayes.joblib",
    "SVM": "SVM.joblib"
}

selected = st.selectbox("Select Model", list(models.keys()))

uploaded = st.file_uploader("Upload test CSV", type="csv")

if uploaded:
    data = pd.read_csv(uploaded)

    if "target" not in data.columns:
        st.error("Target column missing")
        st.stop()

    X = data.drop(columns=["target"])
    y = data["target"]

    model = joblib.load("model/" + models[selected])

    X = X[list(model.feature_names_in_)]

    pred = model.predict(X)
    prob = model.predict_proba(X)

    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(
            y, prob, labels=model.classes_,
            multi_class="ovr", average="macro"
        ),
        "Precision": precision_score(y, pred, average="macro", zero_division=0),
        "Recall": recall_score(y, pred, average="macro", zero_division=0),
        "F1 Score": f1_score(y, pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, pred)
    }

    st.subheader("Evaluation Metrics")
    st.dataframe(pd.DataFrame([metrics]).T)

    st.subheader("Confusion Matrix")
    st.dataframe(pd.DataFrame(
        confusion_matrix(y, pred, labels=model.classes_),
        index=model.classes_,
        columns=model.classes_
    ))

    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(
        classification_report(y, pred, zero_division=0, output_dict=True)
    ).T)

    st.success("Evaluation completed!")
