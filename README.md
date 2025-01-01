# Anomaly Detection in Financial Transactions
This project implements an Anomaly Detection system using the Isolation Forest algorithm to detect unusual or fraudulent transactions in financial datasets. By analyzing patterns in transaction features like amount, frequency, and average transaction value, the model flags potential anomalies for further review, streamlining fraud detection systems.

The application leverages Streamlit for interactive data visualization and model evaluation, offering a user-friendly interface for uploading datasets, visualizing trends, training the model, and testing individual transactions for anomalies.

Project Workflow
Data Upload & Preprocessing:

The user can upload transaction data in CSV format.
Preprocessing steps include calculating basic statistics like mean and standard deviation of transaction amounts and flagging outliers based on these metrics.
Data Visualization:

Interactive charts (like histograms and scatter plots) are generated to help understand the distribution of transaction data and identify patterns or anomalies visually.
Anomaly Detection with Isolation Forest:

The Isolation Forest algorithm is trained to learn the normal behavior of transactions based on features such as:
Transaction_Amount
Average_Transaction_Amount
Frequency_of_Transactions
The trained model is then used to identify anomalies in the dataset.
Model Evaluation:

Performance of the model is evaluated on a test set, and results are displayed in the form of a classification report with metrics such as precision, recall, and F1-score.
User Testing:

The application allows users to input their own transaction details and test if the transaction is flagged as anomalous or normal based on the trained model.
Key Features
Interactive Dashboard: Build and view interactive visualizations with Streamlit, including the ability to visualize transaction distributions, anomalies, and model performance metrics.
Anomaly Detection: Detect fraudulent or suspicious transactions using the Isolation Forest model, which flags outliers based on transaction patterns.
Model Evaluation: Detailed classification reports to evaluate model performance, accessible via a dropdown menu.
Scalability: The system is designed to handle datasets of various sizes, making it adaptable to different use cases in fraud detection.
Project Setup
Prerequisites
Python 3.x (preferably 3.7 or higher)
Required Python libraries are listed in the requirements.txt file.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/AbhishekS31/Anomaly_detection-in-Transactions.git
cd Anomaly_detection-in-Transactions
Create a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run app.py
This will start a local Streamlit server, and the app will be accessible via your browser at http://localhost:8501.

Files and Directories
app.py: Main Streamlit application file for the anomaly detection interface.
model.py: Contains the Isolation Forest model and functions to train and predict anomalies.
utils.py: Helper functions for data processing, visualization, and metrics calculation.
requirements.txt: List of dependencies for the project.
data/: Example datasets (optional).
How It Works
Data Input: The user uploads a CSV file containing transaction data. The data should include at least the following columns:

Transaction_Amount
Average_Transaction_Amount
Frequency_of_Transactions
Data Preprocessing: Basic statistics (mean and standard deviation) of transaction amounts are computed. Anomalies are initially flagged based on a simple threshold (3 standard deviations from the mean) for early detection.

Training the Model: The Isolation Forest algorithm is trained on the dataset, using features like Transaction_Amount, Average_Transaction_Amount, and Frequency_of_Transactions.

Anomaly Detection: Once trained, the model detects anomalies, which are displayed in the output and can be reviewed interactively by the user.

Model Evaluation: After training the model, the classification report is displayed in tabular form, showing metrics like precision, recall, and F1-score.

User Testing: Users can enter transaction details via a form, and the model will predict if the transaction is anomalous (fraudulent) or normal.

Future Improvements
Model Optimization: Tuning the Isolation Forest hyperparameters (e.g., contamination rate) and evaluating different anomaly detection models (e.g., One-Class SVM, Autoencoders).
Handling Imbalanced Data: Implement strategies to handle imbalanced datasets, where fraudulent transactions are rare.
Deployment: Deploy the app as a web service for wider use, allowing real-time anomaly detection for financial systems.
License
This project is licensed under the MIT License – see the LICENSE file for details.