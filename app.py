import streamlit as st
import pandas as pd
import numpy as np
import joblib


@st.cache_resource
def load_model():
    return joblib.load("model_leaf1.joblib")


bundle = load_model()
model = bundle["model"]
columns = bundle["columns"]
communities = sorted(bundle["communities"])

st.title("Dubai Property Price Estimator")
st.write(
    "Pick a community and enter the size and rooms to get an estimated asking price "
    "for a property for sale in Dubai."
)

community = st.selectbox("Community", communities + ["Other (not in the list)"])
size = st.number_input("Size (sqft)", min_value=250, max_value=20000, value=1000, step=50)
bedrooms = st.selectbox(
    "Bedrooms", [0, 1, 2, 3, 4, 5, 6, 7],
    format_func=lambda x: "Studio" if x == 0 else str(x),
)
bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4, 5, 6, 7], index=1)

if st.button("Estimate price"):
    row = pd.DataFrame(0, index=[0], columns=columns)
    row["size_sqft"] = size
    row["bedrooms_n"] = bedrooms
    row["bathrooms_n"] = bathrooms

    key = "Other" if community.startswith("Other") else community
    col = "community_grouped_" + key
    if col in row.columns:
        row[col] = 1

    price = float(np.exp(model.predict(row)[0]))
    st.success(f"Estimated asking price: AED {price:,.0f}")
    st.write(f"Typical range: AED {price * 0.84:,.0f} to AED {price * 1.16:,.0f}")

st.caption(
    "Built from about 4,000 Dubai property listings (August 2024). These are asking prices, "
    "not sold prices. A typical estimate is off by about 16%, and by more for luxury "
    "properties. The model does not know about view, floor, or condition."
)
