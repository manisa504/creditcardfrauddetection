import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Function to load the trained model
@st.cache_data
def load_model(filename):
    return joblib.load(filename)

def preprocess_input(account_number, merchant_id, mcc, 
                     merchant_country, pos_entry_mode, 
                     transaction_amount, available_cash, transaction_date, transaction_time):
    # Format date and time inputs into a single datetime string
    formatted_date = transaction_date.strftime('%Y-%m-%d')
    formatted_time = transaction_time.strftime('%H:%M:%S')
    transaction_datetime_str = f"{formatted_date} {formatted_time}"

    # Convert to datetime object
    transaction_datetime = datetime.strptime(transaction_datetime_str, '%Y-%m-%d %H:%M:%S')

    # Extract day of week and hour of day
    day_of_week = transaction_datetime.weekday()  # Monday=0, Sunday=6
    hour_of_day = transaction_datetime.hour

    # Encode categorical features
    account_number_encoded = LabelEncoder().fit_transform([account_number])
    merchant_id_encoded = LabelEncoder().fit_transform([merchant_id])

    # Handling missing values
    # Replace missing values with a default value
    default_value = 0  # Or any other appropriate default value
    mcc = mcc if mcc is not None else default_value
    merchant_country = merchant_country if merchant_country is not None else default_value
    pos_entry_mode = pos_entry_mode if pos_entry_mode is not None else default_value
    transaction_amount = transaction_amount if transaction_amount is not None else default_value
    available_cash = available_cash if available_cash is not None else default_value

    # Preparing the DataFrame for prediction
    input_data = pd.DataFrame([[account_number_encoded[0], merchant_id_encoded[0], mcc, 
                                merchant_country, pos_entry_mode, 
                                transaction_amount, available_cash, day_of_week, 
                                hour_of_day]],
                              columns=['accountNumber', 'merchantId', 'mcc', 'merchantCountry', 
                                       'posEntryMode', 'transactionAmount', 
                                       'availableCash', 'DayOfWeek', 'HourOfDay'])

    return input_data

def main():
    st.title("Fraud Detection App")

    # Load the model
    model = load_model('fraud_model.pkl')

    # Display an image
    image_url = "https://syndelltech.com/wp-content/uploads/2023/01/fraud-detection-using-machine-ml.png"
    st.image(image_url, caption='Fraud Detection using Machine Learning')

    st.write("Please input the transaction data for fraud prediction.")

    # Create date and time inputs with example default values
    example_date = datetime.today().date()
    example_time = datetime.now().time()
    transaction_date = st.date_input("Transaction Date", value=example_date)
    transaction_time = st.time_input("Transaction Time", value=example_time)

    # Create other input fields
    account_number = st.text_input("Account Number", value="123456")
    merchant_id = st.text_input("Merchant ID", value="A1B2C3")
    mcc = st.number_input("MCC", value=5411, step=1, format='%d')
    merchant_country = st.number_input("Merchant Country", value=840, step=1, format='%d')
    pos_entry_mode = st.number_input("POS Entry Mode", value=81, step=1, format='%d')
    transaction_amount = st.number_input("Transaction Amount", value=100.0, min_value=0.0, format='%f')
    available_cash = st.number_input("Available Cash", value=5000, step=1, format='%d')

    # Button to make prediction
    if st.button("Predict"):
        # Preprocess the input data
        input_data = preprocess_input(account_number, merchant_id, mcc, 
                                      merchant_country, pos_entry_mode, 
                                      transaction_amount, available_cash, 
                                      transaction_date, transaction_time)

        # Make prediction
        prediction = model.predict(input_data)

        # Display the prediction
        st.write("Prediction:", "Fraud" if prediction[0] == 1 else "Not Fraud")

    # Button to load sample fraudulent transaction data
    if st.button("Load Sample Fraudulent Transaction"):
        # Sample fraudulent transaction data
        sample_fraud_data = {
            'account_number': '310',
            'merchant_id': '13245',
            'mcc': 5541,
            'merchant_country': 36,
            'pos_entry_mode': 5,
            'transaction_amount': 10.95,
            'available_cash': 1500,
            'transaction_date': datetime(2023, 1, 1),  # Example date
            'transaction_time': datetime(2023, 1, 1, 3, 0)  # Example time
        }

        # Store the sample data in the session state
        for key, value in sample_fraud_data.items():
            st.session_state[key] = value

        # Display the sample fraudulent transaction data
        st.markdown('---')
        st.markdown('**Sample Fraudulent Transaction Data**')
        st.markdown('---')
        st.markdown(f'**Account Number:** {st.session_state["account_number"]}')
        st.markdown(f'**Merchant ID:** {st.session_state["merchant_id"]}')
        st.markdown(f'**MCC:** {st.session_state["mcc"]}')
        st.markdown(f'**Merchant Country:** {st.session_state["merchant_country"]}')
        st.markdown(f'**POS Entry Mode:** {st.session_state["pos_entry_mode"]}')
        st.markdown(f'**Transaction Amount:** {st.session_state["transaction_amount"]}')
        st.markdown(f'**Available Cash:** {st.session_state["available_cash"]}')
        st.markdown(f'**Transaction Date:** {st.session_state["transaction_date"]}')
        st.markdown(f'**Transaction Time:** {st.session_state["transaction_time"]}')

if __name__ == '__main__':
    main()
