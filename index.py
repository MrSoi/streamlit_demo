import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl

def load_data():
    data = pd.read_csv('loan_dataset.csv')
    df_loan = pd.DataFrame(data)
    st.write(df_loan.head())
    st.bar_chart(df_loan['Education'].value_counts())

def load_prediction():
    income = st.sidebar.slider("ApplicantIncome", min_value=0, max_value=1000000, value=5000)
    loan_amount = st.sidebar.slider("CoapplicantIncome", min_value=0, max_value=400, value=50)

    try:
        model = pkl.load(open('Random_Forest.sav', 'rb'))
        
    except Exception as e:
        st.error(f"Error loading model: {e}")

    if st.sidebar.button("Predict"):
        prediction = model.predict(np.array([[income, loan_amount]]))
        if prediction[0] == 1:
            st.success("Loan Approved")
        else:
            st.error("Loan Denied")

st.title("Loan prediction")
selection = st.sidebar.selectbox("Select the model", ['Load data', 'Predict loan'])
if selection == 'Load data':
    load_data()
elif selection == 'Predict loan':
    load_prediction()