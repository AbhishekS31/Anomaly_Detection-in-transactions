#15) FINAL TESTING ---->
# Relevant features used during training
relevant_features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']

# Get user inputs for features
user_inputs = []
for feature in relevant_features:
user_input = float(input(f"Enter the value for '{feature}': "))
user_inputs.append(user_input)


# Create a DataFrame from user inputs
user_df = pd.DataFrame([user_inputs], columns=relevant_features)

# Predict anomalies using the model
user_anomaly_pred = model.predict(user_df)
s

# Convert the prediction to binary value (0: normal, 1: anomaly)
user_anomaly_pred_binary = 1 if user_anomaly_pred == -1 else 0
if user_anomaly_pred_binary == 1:
print("Anomaly detected: This transaction is flagged as an anomaly.")
else:
print("No anomaly detected: This transaction is normal.")