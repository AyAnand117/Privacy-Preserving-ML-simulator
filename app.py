import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from utils import load_and_preprocess, train_local_model, add_differential_privacy, federated_averaging
from dp_training import train_with_dp

st.set_page_config(page_title = "Privacy-Preserving ML", layout="wide")

# ---------UI------------
st.title(" Privacy-Preserving ML Simulator")


uploaded_file = st.file_uploader("Upload your dataset (creditcard.csv)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else :
    df = pd.read_csv("creditcard.csv")

st.write("Dataset Preview:", df.head())

# Simulate Clients

client1, rest = train_test_split(df, test_size=0.66,random_state=42)
client2, client3 = train_test_split(rest, test_size=0.5,random_state=42)
clients_data = [
    load_and_preprocess(client1),
    load_and_preprocess(client2),
    load_and_preprocess(client3)
    ]

X_test, y_test = load_and_preprocess(df)

# Privacy Level

epsilon = st.slider("Select Privacy Level (Eps)", min_value = 0.1, max_value = 10.0, step = 0.1)

#---------Federated Training------------

if st.button("Run Federated Training with DP"):
    with st.spinner("Training models..."):
        coefs, intercepts = [],[]
        for X,y in clients_data:
            local_coef, local_intercept = train_local_model(X,y)
            dp_coef, dp_intercept = add_differential_privacy(local_coef, local_intercept, epsilon)
            coefs.append(dp_coef)
            intercepts.append(dp_intercept)


        global_coef, global_intercept = federated_averaging(coefs, intercepts)
        model = LogisticRegression()
        model.coef_ = global_coef
        model.intercept_ = global_intercept
        model.classes_ = np.array([0,1])
        y_pred = model.predict(X_test)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.success("Training Complete")

        st.metric("F1 Score", round(f1,3))
        st.metric("Precision Score", round(precision,3))
        st.metric("Recall Score", round(recall,3))


        #Optional : Storing and PLotting Results

        if "results" not in st.session_state:
            st.session_state["results"]=[]
        st.session_state["results"].append((epsilon,f1,precision,recall))


#-----------Performance Chart-----------


if "results" in st.session_state and len(st.session_state["results"])>1:
    st.subheader("Performance vs Privacy")
    results = sorted(st.session_state["results"], key=lambda x:x[0])
    eps,f1s,ps,rs = zip(*results)


    plt.figure(figsize=(10,5))
    plt.plot(eps,f1s, label="F1 Score", marker='o')
    plt.plot(eps,ps, label="Precision Score", marker='s')
    plt.plot(eps,rs, label="Recall Score", marker='^')
    plt.gca().invert_xaxis() # Lower eps = Stronger Privacy
    plt.xlabel("Privacy (Eps)")
    plt.ylabel("Score")
    plt.title("Privacy-Utility Trade-Off")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt.gcf())


st.header("Train with Differential Privacy (Pytorch + Opacus)")
use_dp = st.checkbox("Enable Differential Privacy", value = True)
noise_multiplier = st.slider("Noise Multiplier", min_value = 0.5, max_value = 5.0, step = 1.0)
epochs = st.slider("Epochs", max_value = 20, step = 1, value=5)
batch_size = st.selectbox("Batch Size", [64,128,256], index=2)


if st.button("Run Pytorch DP Training"):
    report, final_eps, eps_curve = train_with_dp(
        data_path="creditcard.csv",
        noise_multiplier=noise_multiplier,
        max_grad_norm=1.0,
        epochs=epochs,
        batch_size=batch_size,
        return_eps_curve = True
        )
    st.text(report)
    st.metric("Final Epsilon", round(final_eps,2))

    st.subheader("Epsilon Progression per Epoch")
    plt.figure(figsize=(8,4))
    plt.plot(range(1,len(eps_curve)+1), eps_curve, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Epsilon")
    plt.grid(True)
    st.pyplot(plt.gcf())


    
