
# Day 7: Preparing and Uploading the Fraud Detection Dataset to PySyft Datasite

Today, I expanded on the environment set up on Day 6 to prepare and upload the **Fraud Detection Dataset** to a **PySyft Datasite**. This process involved not only preserving privacy by creating mock data that mimics the original dataset's structure but also setting the stage for a key workflow: enabling external data scientists to connect to the Datasite. These scientists can explore the mock data—designed to safeguard the real dataset's privacy—and propose experiments based on their findings. Impressively, this allows operations on the real data to be executed and results to be obtained without ever exposing the sensitive information.

---

## **What I Accomplished Today**

1. **Dataset Overview**:
   - Used a smaller sample of the Kaggle Fraud Detection dataset (`small_set.csv`) to prepare for experimentation.
   - Explored key features of the dataset and defined the problem: identifying fraudulent transactions while ensuring data privacy.

2. **Created Mock Data**:
   - Generated mock data from the real dataset by introducing noise and shuffling categories.
   - Ensured the mock data preserves the data structure and class distribution without revealing sensitive patterns.

3. **Uploaded Real and Mock Data to the Datasite**:
   - Organized the dataset into **assets** for features and targets.
   - Packaged the assets into a `sy.Dataset` object with metadata.
   - Uploaded the dataset to the PySyft Datasite for experimentation.

---
### **Preprocessing the Dataset**

To prepare the dataset for analysis and ensure privacy, I removed personally identifiable information (PII) and transformed categorical variables into numeric representations. Here's the concise preprocessing pipeline:

```python
import pandas as pd

# Preprocessing function
def preprocess_data(df):
    # Drop PII and irrelevant columns
    df = df.drop(columns=[
        "Unnamed: 0", "trans_date_trans_time", "cc_num", 
        "first", "last", "street", "city", "state", "zip", 
        "unix_time", "dob", "trans_num", "job", 
        "lat", "long", "merch_lat", "merch_long"
    ])
    
    # Encode categorical variables
    df["merchant"] = df["merchant"].astype("category").cat.codes
    df["category"] = df["category"].astype("category").cat.codes
    df["gender"] = df["gender"].map({"M": 0, "F": 1}).fillna(-1)
    
    return df

# Processing and saving the data
def process_and_save_data(input_path, output_path):
    # Load and preprocess the dataset
    data = preprocess_data(pd.read_csv(input_path))
    # Save the processed dataset
    data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

# Example usage
process_and_save_data("small_set.csv", "processed_data.csv")
````

### **Key Changes in Preprocessing**

1. **Removed PII**: Columns like `first`, `last`, `street`, and `dob` were dropped to prevent reidentification.
2. **Dropped Location Coordinates**: Latitude and longitude columns (`lat`, `long`, `merch_lat`, `merch_long`) were excluded to mitigate privacy risks.
3. **Categorical Encoding**:
    - `merchant` and `category` were encoded numerically using category codes.
    - `gender` was mapped to binary values (`M` as 0, `F` as 1) with missing values filled as -1.

This processed dataset retains the essential information for analysis while minimizing privacy concerns.
## **Exploring the Dataset**

Before uploading the dataset to the PySyft Datasite, I analyzed the processed data to understand its structure and key features. This step ensures the dataset is suitable for training models while protecting sensitive information.

### **Loading and Inspecting the Dataset**
The dataset used here is a preprocessed version of the Kaggle Fraud Detection dataset (`processed_data.csv`), which has been cleaned to remove PII and encode categorical variables.

```python
import pandas as pd

# Load the processed dataset
data = pd.read_csv("processed_data.csv")

# Display the first few rows
print(data.head())

# Check the dataset structure
print(data.info())

# Basic statistics
print(data.describe())
````

### **Sample Data**

|merchant|category|amt|gender|city_pop|is_fraud|
|---|---|---|---|---|---|
|406|5|11.80|1|302|0|
|499|0|98.43|0|673342|0|
|46|6|97.94|1|520|0|
|501|13|9.90|0|206|0|
|79|8|89.62|1|3343|0|

### **Key Features**

- **merchant**: Encoded merchant identifier (numeric).
- **category**: Encoded merchant category (numeric).
- **amt**: Transaction amount (numeric).
- **gender**: Encoded gender (0 for Male, 1 for Female, -1 for missing values).
- **city_pop**: Population of the customer's city (numeric).
- **is_fraud**: Target variable indicating fraudulent transactions (binary).

### **Initial Observations**

- **amt** has a wide range, suggesting a need for normalization in model training.
- **category** and **merchant** are already encoded and ready for use in models.
- **is_fraud** is imbalanced, requiring techniques to address this imbalance.

---

## **Dataset Preparation**

### **Real Data**

The processed dataset (`processed_data.csv`) was derived from the Kaggle Fraud Detection Dataset, ensuring it is anonymized and ready for secure analysis. Key features like transaction amounts, merchant categories, and city population were preserved.

### **Mock Data**

To maintain privacy, mock data was generated to simulate the real dataset while ensuring no private information is leaked. Here's how the mock data was created:

```python
import numpy as np

# Fix seed for reproducibility
np.random.seed(12345)

# Create mock data
data_mock = data.copy()
data_mock['amt'] += np.random.uniform(size=len(data))
data_mock['category'] = data_mock['category'].sample(frac=1).reset_index(drop=True)
```

---

## **Uploading the Dataset**

### **Step 1: Create Assets**

Real and mock data were organized into `sy.Asset` objects for uploading:

```python
import syft as sy

features_asset = sy.Asset(
    name="Fraud Detection Data: Features",
    data=data.drop(columns=['is_fraud']),       # Real data
    mock=data_mock.drop(columns=['is_fraud'])   # Mock data
)

targets_asset = sy.Asset(
    name="Fraud Detection Data: Targets",
    data=data['is_fraud'],       # Real targets
    mock=data_mock['is_fraud']   # Mock targets
)
```

### **Step 2: Create and Describe the Dataset**

The assets were added to a `sy.Dataset` object with metadata:

```python
fraud_dataset = sy.Dataset(
    name="Fraud Detection Dataset",
    description="A dataset for identifying fraudulent transactions.",
    summary="This dataset contains transaction features and targets for fraud detection.",
    citation="Kaggle Fraud Detection Dataset",
    url="https://www.kaggle.com/datasets/kartik2112/fraud-detection"
)

fraud_dataset.add_asset(features_asset)
fraud_dataset.add_asset(targets_asset)
```

### **Step 3: Upload the Dataset**

Finally, the dataset was uploaded to the PySyft Datasite:

```python
client.upload_dataset(dataset=fraud_dataset)
```

---

## **Next Steps**

1. Understand the workflow of an external data scientist connecting to the Datasite.  
2. Explore the mock dataset, as the real data remains inaccessible to preserve privacy.  
3. Propose and execute experiments using the mock data to simulate operations on the real dataset.  
4. Leverage Differential Privacy techniques to ensure the results maintain data security.  
5. Train models using Federated Learning and evaluate their performance based on the results obtained from the mock-to-real data workflow.

