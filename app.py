import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import datetime

# --- 1. Load Model and Preprocessor ---

# Use st.cache_resource to load the model and preprocessor only once
@st.cache_resource
def load_assets():
    """Loads the saved Keras model and scikit-learn preprocessor."""
    try:
        model = load_model('csat_ann_model.h5')
        preprocessor = joblib.load('csat_preprocessor.joblib')
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model or preprocessor: {e}")
        return None, None

model, preprocessor = load_assets()

# --- 2. Set Up The Web App Interface ---

st.title("🤖 E-Commerce CSAT Score Predictor")
st.write("Enter the details of a customer support interaction to predict the CSAT score (1-5).")

# Create a sidebar for user inputs
st.sidebar.header("Enter Interaction Details:")

# --- 3. Get User Input ---
# We need to get input for all the features our preprocessor expects.
# We must re-create the feature lists from our notebook.

# Categorical Features
channel_name = st.sidebar.selectbox("Channel Name", ['Inbound', 'Outcall', 'Chat', 'Email'])
category = st.sidebar.selectbox("Category", ['Order Related', 'Product Queries', 'Refund Related', 'Feedback', 'Others'])
sub_category = st.sidebar.selectbox("Sub-category", ['Order status enquiry', 'Product Specific Information', 'Refund Enquiry', 'Seller Cancelled Order', 'Life Insurance', 'General Information', 'Others'])
customer_city = st.sidebar.text_input("Customer City", "Mumbai") # Free text
product_category = st.sidebar.selectbox("Product Category", ['Mobile', 'Electronics', 'Apparel', 'Grocery', 'Home', 'Life Insurance', 'Others'])
tenure_bucket = st.sidebar.selectbox("Agent Tenure Bucket", ['On Job Training', '>90', '0-30', '31-60', '61-90'])
agent_shift = st.sidebar.selectbox("Agent Shift", ['Morning', 'Evening', 'Night'])

# Numerical Features
item_price = st.sidebar.number_input("Item Price", min_value=0.0, value=1500.0, step=100.0)
connected_handling_time = st.sidebar.number_input("Handling Time (seconds)", min_value=0.0, value=300.0, step=30.0)

# Datetime Features (for Engineering)
# We ask for the datetimes and then engineer the same features as in the notebook
issue_reported_at = st.sidebar.datetime_input("Issue Reported At", datetime.datetime.now() - datetime.timedelta(minutes=30))
issue_responded = st.sidebar.datetime_input("Issue Responded At", datetime.datetime.now())


# --- 4. Process Input and Predict ---

# Create a button to make the prediction
if st.button("Predict CSAT Score"):
    if model is None or preprocessor is None:
        st.error("Model assets not loaded. Please check file paths and restart.")
    else:
        try:
            # --- 4a. Perform Feature Engineering (Same as notebook) ---
            
            # 1. Calculate response time
            response_time = (issue_responded - issue_reported_at)
            response_time_in_minutes = response_time.total_seconds() / 60
            
            # Check for negative response time
            if response_time_in_minutes < 0:
                st.error("Error: 'Issue Responded At' cannot be before 'Issue Reported At'.")
            else:
                # 2. Extract hour
                issue_reported_hour = issue_reported_at.hour
                
                # 3. Extract day of week
                issue_reported_dayofweek = issue_reported_at.weekday() # Monday=0, Sunday=6

                # --- 4b. Create Input DataFrame ---
                # The column names MUST match the training data
                
                input_data = {
                    'channel_name': [channel_name],
                    'category': [category],
                    'Sub-category': [sub_category],
                    'Customer_City': [customer_city],
                    'Product_category': [product_category],
                    'Item_price': [item_price],
                    'connected_handling_time': [connected_handling_time],
                    'Tenure Bucket': [tenure_bucket],
                    'Agent Shift': [agent_shift],
                    'response_time_in_minutes': [response_time_in_minutes],
                    'issue_reported_hour': [issue_reported_hour],
                    'issue_reported_dayofweek': [issue_reported_dayofweek]
                }
                
                input_df = pd.DataFrame(input_data)
                
                st.subheader("Raw Input for Model:")
                st.dataframe(input_df)

                # --- 4c. Transform and Predict ---
                
                # Transform using the loaded preprocessor
                input_processed = preprocessor.transform(input_df)
                
                # Predict using the loaded model
                prediction_probs = model.predict(input_processed)
                
                # Get the class with the highest probability
                prediction_index = np.argmax(prediction_probs, axis=1)[0]
                
                # Convert 0-4 index back to 1-5 CSAT Score
                predicted_csat_score = prediction_index + 1

                # --- 4d. Display Results ---
                
                st.subheader(f"Predicted Customer Satisfaction (CSAT) Score:")
                
                # Display the score with a custom color
                if predicted_csat_score in [1, 2]:
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{predicted_csat_score}</h1>", unsafe_allow_html=True)
                    st.warning("Prediction: Very Dissatisfied. Action may be required.")
                elif predicted_csat_score == 3:
                    st.markdown(f"<h1 style='text-align: center; color: orange;'>{predicted_csat_score}</h1>", unsafe_allow_html=True)
                    st.info("Prediction: Neutral.")
                else: # 4 or 5
                    st.markdown(f"<h1 style='text-align: center; color: green;'>{predicted_csat_score}</h1>", unsafe_allow_html=True)
                    st.success("Prediction: Satisfied!")

                # Display the prediction probabilities
                st.subheader("Prediction Probabilities for Each Score (1-5):")
                prob_df = pd.DataFrame(
                    prediction_probs[0],
                    index=[f'CSAT {i+1}' for i in range(5)],
                    columns=['Probability']
                )
                st.bar_chart(prob_df)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
