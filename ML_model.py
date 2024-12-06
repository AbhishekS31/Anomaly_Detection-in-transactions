# 13)MACHINE LEARNING MODEL FOR DETECTING ANOMALIES USING ISOLATION FOREST ALGO
relevant_features = ['Transaction_Amount',
                    'Average_Transaction_Amount',
                    'Frequency_of_Transactions']

# 13-A)Split data into features (X) and target variable (y)
X = data[relevant_features]
y = data['Is_Anomaly']

# 13-B)Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 13-C)Train the Isolation Forest model
model = IsolationForest(contamination=0.02, random_state=42)
model.fit(X_train)