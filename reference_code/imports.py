# following libraries required
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report


# Specify the correct file path
file_path = "C:\\Users\\Abhishek\\Downloads\\transaction_anomalies_dataset.csv"


# Read the CSV file
data = pd.read_csv(file_path)
print(data.head())