import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import streamlit as st
from sklearn.preprocessing import StandardScaler

# 1) Load the dataset (Streamlit File Uploader)
st.title("Transaction Anomaly Detection")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file into a pandas DataFrame
    data = pd.read_csv(uploaded_file)

    # 2) Calculate mean and standard deviation of Transaction Amount
    mean_amount = data['Transaction_Amount'].mean()
    std_amount = data['Transaction_Amount'].std()

    # 3) Define the anomaly threshold (3 standard deviations from the mean)
    anomaly_threshold = mean_amount + 3 * std_amount

    # 4) Flag anomalies based on new threshold (more lenient flagging)
    data['Is_Anomaly'] = data['Transaction_Amount'] > anomaly_threshold

    # 5) Calculate the number of anomalies
    num_anomalies = data['Is_Anomaly'].sum()

    # 6) Calculate the total number of instances in the dataset
    total_instances = data.shape[0]

    # 7) Calculate the ratio of anomalies
    anomaly_ratio = num_anomalies / total_instances
    st.write(f"Anomaly Ratio: {anomaly_ratio:.4f}")

    # 8) Machine Learning Model for detecting anomalies using Isolation Forest
    relevant_features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']

    # 9) Split data into features (X) and target variable (y)
    X = data[relevant_features]
    y = data['Is_Anomaly']

    # 10) Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[relevant_features])

    # 11) Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 12) Train the Isolation Forest model (with adjusted contamination rate)
    model = IsolationForest(contamination=0.01, random_state=42)  # Adjusted contamination to 1%
    model.fit(X_train)

    # 13) Performance Analysis
    # Predict anomalies on the test set
    y_pred = model.predict(X_test)

    # Convert predictions to binary values (1: normal, 0: anomaly)
    y_pred_binary = [1 if pred == 1 else 0 for pred in y_pred]

    # 14) Display classification report in tabular form
    report = classification_report(y_test, y_pred_binary, target_names=['Normal', 'Anomaly'], output_dict=True)

    # Create a DataFrame from the classification report
    report_df = pd.DataFrame(report).transpose()

    # Add a dropdown to allow the user to view the classification report
    st.subheader("Model Performance Summary")
    show_report = st.selectbox("Would you like to see the model's classification report?", ["No", "Yes"])

    if show_report == "Yes":
        st.write("### Classification Report")
        st.dataframe(report_df)

    # 15) Final Testing (User Input)
    st.subheader("Enter Transaction Details to Check for Anomaly")

    # Get user inputs for features (streamlit widgets)
    user_inputs = []
    for feature in relevant_features:
        user_input = st.number_input(f"Enter the value for '{feature}':", min_value=0.0, step=0.1)
        user_inputs.append(user_input)

    # Create a DataFrame from user inputs
    user_df = pd.DataFrame([user_inputs], columns=relevant_features)

    # Predict anomalies using the model
    if st.button('Check for Anomaly'):
        user_df_scaled = scaler.transform(user_df)  # Apply scaling to user input
        user_anomaly_pred = model.predict(user_df_scaled)

        # Convert the prediction to binary value (0: normal, 1: anomaly)
        user_anomaly_pred_binary = 1 if user_anomaly_pred == -1 else 0

        if user_anomaly_pred_binary == 1:
            st.error("Anomaly detected: This transaction is flagged as an anomaly.")
        else:
            st.success("No anomaly detected: This transaction is normal.")
