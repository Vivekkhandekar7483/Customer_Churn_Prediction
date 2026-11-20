import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "customer_churn_best_model.pkl")
FEATURES = ["Age","Gender","Tenure","Usage Frequency","Support Calls","Payment Delay",
            "Subscription Type","Contract Length","Total Spend","Last Interaction"]

@st.cache_resource
def load_model(path):
    # Load only a trusted model artifact.
    return joblib.load(path)

st.title("Customer Churn Prediction")
st.write("Educational prototype. Enter customer details to obtain a model estimate.")
if not os.path.exists(MODEL_PATH):
    st.error("Put customer_churn_best_model.pkl in the same folder as app.py.")
    st.stop()
try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()

with st.form("customer_form"):
    age=st.number_input("Age",0,120,30)
    gender=st.selectbox("Gender",["Female","Male","Other"])
    tenure=st.number_input("Tenure",0,10000,12)
    usage=st.number_input("Usage Frequency",0,10000,10)
    calls=st.number_input("Support Calls",0,10000,2)
    delay=st.number_input("Payment Delay",0,10000,5)
    sub=st.selectbox("Subscription Type",["Basic","Standard","Premium"])
    contract=st.selectbox("Contract Length",["Monthly","Quarterly","Annual"])
    spend=st.number_input("Total Spend",0.0,100000000.0,500.0)
    last=st.number_input("Last Interaction",0,10000,10)
    submitted=st.form_submit_button("Predict churn")

if submitted:
    X=pd.DataFrame([{"Age":age,"Gender":gender,"Tenure":tenure,"Usage Frequency":usage,
      "Support Calls":calls,"Payment Delay":delay,"Subscription Type":sub,
      "Contract Length":contract,"Total Spend":spend,"Last Interaction":last}],columns=FEATURES)
    try:
        pred=int(model.predict(X)[0])
        if pred==1: st.error("Predicted outcome: Customer is likely to CHURN.")
        else: st.success("Predicted outcome: Customer is likely to STAY.")
        if hasattr(model,"predict_proba"):
            probabilities=model.predict_proba(X)[0]
            classes=list(model.classes_)
            if 1 in classes:
                p=float(probabilities[classes.index(1)])
                st.metric("Estimated churn probability",f"{p:.1%}")
                st.progress(max(0.0,min(1.0,p)))
        with st.expander("View submitted values"): st.dataframe(X,use_container_width=True)
        st.caption("This is an estimate, not a guarantee. Independent validation is pending.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info("Check feature names, category labels and compatible package versions.")
st.divider()
st.caption("Educational prototype — not validated for production decisions.")
