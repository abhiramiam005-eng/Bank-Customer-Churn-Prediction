
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("churn_model.pkl")

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Bank Customer Churn Prediction")
st.write("Enter customer details to predict the possibility of churn.")

st.subheader("Customer Information")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=45)
        gender = st.selectbox("Gender", ["M", "F"])
        dependent_count = st.number_input("Dependent Count", min_value=0, max_value=10, value=2)
        education = st.selectbox(
            "Education Level",
            ["Unknown", "Uneducated", "High School", "College",
             "Graduate", "Post-Graduate", "Doctorate"]
        )
        marital_status = st.selectbox(
            "Marital Status",
            ["Unknown", "Single", "Married", "Divorced"]
        )
        income = st.selectbox(
            "Income Category",
            ["Unknown", "Less than $40K", "$40K - $60K",
             "$60K - $80K", "$80K - $120K", "$120K +"]
        )
        card_category = st.selectbox(
            "Card Category",
            ["Blue", "Silver", "Gold", "Platinum"]
        )
        months_on_book = st.number_input("Months on Book", min_value=1, max_value=100, value=36)
        relationship_count = st.number_input("Total Relationship Count", min_value=1, max_value=10, value=4)

    with col2:
        inactive_months = st.number_input("Months Inactive (Last 12 Months)", min_value=0, max_value=12, value=2)
        contacts = st.number_input("Contacts Count (Last 12 Months)", min_value=0, max_value=20, value=2)
        credit_limit = st.number_input("Credit Limit", min_value=0.0, value=5000.0)
        revolving_balance = st.number_input("Total Revolving Balance", min_value=0.0, value=1000.0)
        amount_change = st.number_input("Total Amount Change Q4-Q1", min_value=0.0, value=0.7)
        total_trans_amount = st.number_input("Total Transaction Amount", min_value=0.0, value=4000.0)
        total_trans_count = st.number_input("Total Transaction Count", min_value=1, value=60)
        count_change = st.number_input("Total Count Change Q4-Q1", min_value=0.0, value=0.7)
        utilization = st.number_input("Average Utilization Ratio", min_value=0.0, max_value=1.0, value=0.2)

    submitted = st.form_submit_button("Predict Churn")

if submitted:

    input_data = pd.DataFrame([{
        "Customer_Age": customer_age,
        "Gender": gender,
        "Dependent_count": dependent_count,
        "Education_Level": education,
        "Marital_Status": marital_status,
        "Income_Category": income,
        "Card_Category": card_category,
        "Months_on_book": months_on_book,
        "Total_Relationship_Count": relationship_count,
        "Months_Inactive_12_mon": inactive_months,
        "Contacts_Count_12_mon": contacts,
        "Credit_Limit": credit_limit,
        "Total_Revolving_Bal": revolving_balance,
        "Total_Amt_Chng_Q4_Q1": amount_change,
        "Total_Trans_Amt": total_trans_amount,
        "Total_Trans_Ct": total_trans_count,
        "Total_Ct_Chng_Q4_Q1": count_change,
        "Avg_Utilization_Ratio": utilization
    }])

    # Feature engineering used during model training
    input_data["Average_Transaction_Amount"] = (
        input_data["Total_Trans_Amt"] / input_data["Total_Trans_Ct"]
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")

    st.write(f"Churn Probability: **{probability:.2%}**")
