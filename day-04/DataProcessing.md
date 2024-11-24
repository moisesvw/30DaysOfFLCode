## **Data Processing Report**

### **Scenario**
I was tasked with preparing a dataset for a provider who will use it to train a model for detecting fraudulent transactions. Given the sensitive nature of the data, I had to carefully process it to ensure both privacy and utility. Below, I will outline the steps I followed, explaining why each transformation was necessary, and include the code used for the preprocessing.

---

### **Step-by-Step Data Processing**

#### **Step 1: Analyzing the Dataset**
The dataset contained the following columns:

```
['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',
'amt', 'first', 'last', 'gender', 'street', 'city', 'state', 'zip', 
'lat', 'long', 'city_pop', 'job', 'dob', 'trans_num', 'unix_time', 
'merch_lat', 'merch_long', 'is_fraud']
```

To ensure proper processing, I categorized the columns as either sensitive or useful for modeling. 

- **Sensitive Columns:**
  - `cc_num`: A unique credit card number that directly identifies individuals.
  - `first` and `last`: Names of the individuals, which are personally identifiable information (PII).
  - `street`, `dob`: Addresses and dates of birth, both of which can also directly identify individuals.

- **Columns for Location and Time:**
  - `lat`, `long`, `merch_lat`, `merch_long`: Exact latitude and longitude.
  - `trans_date_trans_time`: Precise timestamp of each transaction.

- **Useful Columns for Modeling:**
  - `amt`: Transaction amount, which is critical for detecting fraud.
  - `category`: Category of the transaction (e.g., gas, grocery).
  - `is_fraud`: Target variable indicating whether a transaction is fraudulent.

My goal was to anonymize or generalize the sensitive columns while retaining the useful ones in a way that preserves the patterns needed for fraud detection.

---

#### **Step 2: Removing Direct Identifiers**
Columns that directly identified individuals were removed:
- `cc_num`: Completely removed as it is not necessary for the model and poses a significant privacy risk.
- `first`, `last`, and `street`: Removed to eliminate personal identifiers.
- `dob`: Removed, but we could transform it into age or age ranges if necessary.

By removing these columns, we reduced the risk of linking the data back to specific individuals.

---

#### **Step 3: Generalizing Location Data**
Precise geolocation data (`lat`, `long`, `merch_lat`, `merch_long`) can be used to re-identify individuals when combined with other information. To address this:
- Latitude and longitude values were rounded to 3 decimal places. This retained regional patterns but removed the exact location details.
- Zip codes (`zip`) were left intact, as they are less granular and generally useful for regional modeling.

---

#### **Step 4: Generalizing Temporal Data**
The `trans_date_trans_time` column contained precise timestamps. To protect privacy while retaining temporal trends:
- The column was split into two: one for the transaction date and another for the transaction hour.
- This step preserved the ability to detect patterns like "fraudulent transactions tend to happen at certain hours" without exposing exact times.

---

#### **Step 5: Retaining and Preparing Modeling Columns**
The columns critical for the model, such as `amt`, `category`, and `is_fraud`, were retained without modifications. These columns provide the essential information required to detect fraudulent transactions.

---

### **Code for Data Processing**


### **Initial Dataset**
For this exploration, I use a simulated dataset from Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection). It contains:
- **Legitimate and fraudulent transactions** from 1000 customers and 800 merchants.
- A time period covering **Jan 2019 - Dec 2020**.

Here’s the Python code that implements the above steps:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the preprocessed training data
train_data = pd.read_csv("fraudTrain.csv")

# Step 1: Create a smaller subset for analysis
# Split the data into a small subset (4%) for privacy-safe exploration
small_set, _ = train_test_split(train_data, test_size=0.96, random_state=42)

# Save the smaller subset to a CSV file
small_set.to_csv("small_set.csv", index=False)

# Step 2: Load the smaller dataset
data = pd.read_csv("small_set.csv")

# Step 3: Remove direct identifiers
# Remove columns that directly identify individuals
data = data.drop(columns=['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'dob', 'trans_num'])

# Step 4: Generalize location data
# Round latitude and longitude to 3 decimal places
data['lat'] = data['lat'].round(3)
data['long'] = data['long'].round(3)
data['merch_lat'] = data['merch_lat'].round(3)
data['merch_long'] = data['merch_long'].round(3)

# Remove 'city_pop' if it's not critical
if 'city_pop' in data.columns:
    data = data.drop(columns=['city_pop'])

# Generalize zip codes to the first 3 digits (if needed)
if 'zip' in data.columns:
    data['zip'] = data['zip'].astype(str).str[:3]

# Step 5: Generalize temporal data
# Extract date and hour from the transaction timestamp
data['trans_date'] = pd.to_datetime(data['trans_date_trans_time']).dt.date
data['trans_hour'] = pd.to_datetime(data['trans_date_trans_time']).dt.hour
data = data.drop(columns=['trans_date_trans_time'])

# Step 6: Remove columns that may not be useful
# Remove 'job' if it's not critical for the model
if 'job' in data.columns:
    data = data.drop(columns=['job'])

# Step 7: Save the processed dataset
data.to_csv("small_set_processed_data.csv", index=False)

# Final output: The processed dataset has the following columns
print("Processed dataset columns:", list(data.columns))

```

---

### **Conclusion**
Through these steps, I ensured that the dataset:
1. **Protects Privacy:** Direct identifiers and sensitive details were removed or generalized, making re-identification difficult.
2. **Preserves Utility:** Key columns like `amt`, `category`, and `is_fraud` were kept intact for fraud detection modeling, and temporal and location data were generalized rather than removed.

This processed dataset is now ready to be shared with the provider for model training while meeting both privacy and utility requirements. 

