# Anomaly Detection in Financial Transactions

This project implements an **Anomaly Detection** system using the **Isolation Forest** algorithm to detect unusual or fraudulent transactions in financial datasets. By analyzing patterns in transaction features like amount, frequency, and average transaction value, the model flags potential anomalies for further review, streamlining fraud detection systems.

The application leverages **Streamlit** for interactive data visualization and model evaluation, offering a user-friendly interface for uploading datasets, visualizing trends, training the model, and testing individual transactions for anomalies.

You can try the live demo of the project here:  
**[Live Project](https://anomaly-detetction.streamlit.app/)**

Sample Dataset:
**Dataset**: [sample_transactions.csv](https://github.com/AbhishekS31/Anomaly_detection-in-Transactions/blob/main/transaction_anomalies_dataset.csv)
(Note : you can use a different dataset with similar parameters)

## Project Workflow

### 1. Data Upload & Preprocessing:
- Upload transaction data in CSV format.
- Compute basic statistics (mean, standard deviation) of transaction amounts.
- Flag outliers based on these metrics.

### 2. Data Visualization:
- Generate interactive charts (e.g., histograms, scatter plots) to explore transaction data and detect potential anomalies visually.

### 3. Anomaly Detection with Isolation Forest:
- Train the **Isolation Forest** model using features like:
   - `Transaction_Amount`
   - `Average_Transaction_Amount`
   - `Frequency_of_Transactions`
- Detect anomalies in the dataset.

### 4. Model Evaluation:
- Evaluate model performance using precision, recall, and F1-score, and display a classification report

### 5. User Testing:
- Users can input their own transaction details to check if they are flagged as anomalous or normal.

## Key Features
- **Interactive Dashboard**: View visualizations of transaction distributions, anomalies, and model performance metrics with Streamlit.
- **Anomaly Detection**: Detect fraudulent transactions using the Isolation Forest model.
- **Model Evaluation**: Access detailed classification reports to evaluate the model's performance.
- **Scalability**: The model can handle datasets of various sizes, adaptable to different fraud detection needs.


### Prerequisites
- Python 3.x (preferably 3.7 or higher)
- The required Python libraries are listed in the `requirements.txt` file.

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/AbhishekS31/Anomaly_detection-in-Transactions.git
    cd Anomaly_detection-in-Transactions
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate   # On macOS/Linux
    venv\Scripts\activate      # On Windows
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
    This will start a local Streamlit server, and the app will be accessible via your browser at `http://localhost:8501`.

## Files and Directories
- **`app.py`**: Main Streamlit application file for the anomaly detection interface.
- **`model.py`**: Contains the Isolation Forest model and functions to train and predict anomalies.
- **`utils.py`**: Helper functions for data processing, visualization, and metrics calculation.
- **`requirements.txt`**: List of dependencies for the project.
- **`data/`**: Example datasets (optional).

## How It Works

### 1. Data Input
Upload a CSV file containing transaction data with at least these columns:
- `Transaction_Amount`
- `Average_Transaction_Amount`
- `Frequency_of_Transactions`

### 2. Data Preprocessing
- Compute basic statistics (mean and standard deviation) of transaction amounts.
- Flag anomalies based on a simple threshold (3 standard deviations from the mean).

### 3. Training the Model
Train the Isolation Forest model using the features `Transaction_Amount`, `Average_Transaction_Amount`, and `Frequency_of_Transactions`.

### 4. Anomaly Detection
Once trained, the model identifies anomalies, which are displayed interactively.

### 5. Model Evaluation
The classification report with precision, recall, and F1-score metrics is shown to evaluate model performance.

### 6. User Testing
Users can input transaction details via a form, and the model will predict whether the transaction is anomalous (fraudulent) or normal.

## Future Improvements
- **Model Optimization**: Tune the Isolation Forest hyperparameters (e.g., contamination rate) and explore other anomaly detection models (e.g., One-Class SVM, Autoencoders).
- **Handling Imbalanced Data**: Implement strategies for dealing with imbalanced datasets where fraudulent transactions are rare.
- **Deployment**: Deploy the app as a web service for real-time anomaly detection in financial systems.

## License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
