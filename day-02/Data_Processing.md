# Data Processing for Credit Card Fraud Detection

In this document, I outline the steps I take to preprocess and partition the dataset for use in Federated Learning.

---

## **1. Preprocessing the Dataset**

### **Initial Dataset**
The original dataset contains detailed transaction data, including customer information and geographic details. To optimize training, I preprocess the data by:
- Dropping irrelevant columns that don’t contribute to the prediction task.
- Encoding categorical features into numerical representations.
- Calculating the geographic distance between the customer and the merchant.

### **Preprocessing Script**
Here’s the Python script I use to preprocess the data:

```python
import pandas as pd
from geopy.distance import geodesic

def preprocess_data(df):
    df = df.drop(columns=["Unnamed: 0", "trans_date_trans_time", "cc_num", 
                          "first", "last", "street", "city", "state", "zip", 
                          "unix_time", "dob", "trans_num", "job"])
    df['distance'] = df.apply(
        lambda row: geodesic((row['lat'], row['long']), (row['merch_lat'], row['merch_long'])).km,
        axis=1
    )
    df = df.drop(columns=["lat", "long", "merch_lat", "merch_long"])
    df["merchant"] = df["merchant"].astype("category").cat.codes
    df["category"] = df["category"].astype("category").cat.codes
    df["gender"] = df["gender"].map({"M": 0, "F": 1}).fillna(-1)
    return df

def process_and_save_data(train_path, test_path, train_output, test_output):
    train_data = preprocess_data(pd.read_csv(train_path))
    test_data = preprocess_data(pd.read_csv(test_path))
    train_data.to_csv(train_output, index=False)
    test_data.to_csv(test_output, index=False)
    print(f"Processed data saved to {train_output} and {test_output}")

if __name__ == "__main__":
    process_and_save_data("fraudTrain.csv", "fraudTest.csv", 
                          "fraudTrain_preprocessed.csv", "fraudTest_preprocessed.csv")
```

This script ensures the dataset is clean, consistent, and ready for distributed training in a Federated Learning framework.


## **2. Partitioning the Dataset**

### **Goal**
My goal here is to split the preprocessed training data into three subsets to simulate private datasets, as if they belong to different financial institutions.

### **Partitioning Script**
Here's the script I use to divide the data:

```python
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the preprocessed training data
train_data = pd.read_csv("fraudTrain_preprocessed.csv")

# Split the data into three subsets
subset_1, temp = train_test_split(train_data, test_size=0.67, random_state=42)
subset_2, subset_3 = train_test_split(temp, test_size=0.5, random_state=42)

# Save each subset to a separate CSV file
subset_1.to_csv("subset_1.csv", index=False)
subset_2.to_csv("subset_2.csv", index=False)
subset_3.to_csv("subset_3.csv", index=False)
```

This script ensures that the data is evenly and randomly divided, maintaining representativeness in each subset for use in Federated Learning scenarios.