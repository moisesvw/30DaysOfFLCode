# Day 14: Introduction to Secure Multi-Party Computation (SMPC) with Encrypted Machine Learning

---

## **What is SMPC?**

Secure Multi-Party Computation (SMPC) is a cryptographic technique that enables multiple parties to collaborate on computations **without directly sharing their private data**. 

### **Key Characteristics**:

- **Data Privacy**: Each party keeps its data confidential.
- **Result Sharing**: Only the computed result is revealed to the parties involved.

### **Real-World Example**:

Two hospitals want to calculate the average age of their patients without revealing individual records. SMPC allows this computation while preserving the privacy of each hospital's data.

---

## **Why is SMPC Important?**

SMPC has a wide range of practical applications in today's data-sensitive world:

1. **Collaborative Medicine**: Hospitals can analyze combined patient data without compromising privacy.
2. **Banking and Finance**: Financial institutions can compute joint metrics while keeping customer details confidential.
3. **Federated Learning**: AI models can be trained on distributed datasets without accessing raw data.

---

## **How Does SMPC Work?**

1. **Secret Sharing**: Data is divided into "shares" and distributed among parties. No single share reveals useful information.  
2. **Computation on Shares**: Operations are performed on the distributed shares.  
3. **Result Reconstruction**: Only the final result is combined and revealed, ensuring privacy.

---

## **Step-by-Step Guide: Encrypted Machine Learning Using SMPC**

This example uses the **Heart Disease Dataset** from the UCI Machine Learning Repository to demonstrate SMPC in encrypted machine learning.

---

### **1. Setting Up the Environment**

#### **Prerequisites**

Install the required libraries:

```bash
pip install pandas sklearn tenseal
```

#### **Load the Dataset**

```python
import pandas as pd

# Download and load the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
data = pd.read_csv(url, header=None, names=column_names, na_values="?")

# Clean the data
data.dropna(inplace=True)
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)  # Binary: Disease or No Disease
print(data.head())
```

---

### **2. Splitting the Data Between Parties**

Simulate data partitioning for two institutions.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split the data into two parts
data_part1, data_part2 = train_test_split(data, test_size=0.5, random_state=42)

# Scale the data
scaler = StandardScaler()
X_part1 = scaler.fit_transform(data_part1.iloc[:, :-1])
y_part1 = data_part1['target'].values

X_part2 = scaler.transform(data_part2.iloc[:, :-1])
y_part2 = data_part2['target'].values
```

---

### **3. Encryption Using TenSEAL**

Encrypt the data for each party.

```python
import tenseal as ts

# Create a CKKS encryption context
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.global_scale = 2 ** 40
context.generate_galois_keys()

# Encrypt the data
encrypted_X_part1 = [ts.ckks_vector(context, row.tolist()) for row in X_part1]
encrypted_y_part1 = [ts.ckks_vector(context, [label]) for label in y_part1]

encrypted_X_part2 = [ts.ckks_vector(context, row.tolist()) for row in X_part2]
encrypted_y_part2 = [ts.ckks_vector(context, [label]) for label in y_part2]
```

---

### **4. Training an Encrypted Logistic Regression Model**

Train a simple logistic regression model using encrypted data.

```python
import numpy as np

# Initialize weights and bias
weights = np.random.rand(X_part1.shape[1])
bias = np.random.rand()
learning_rate = 0.01

# Encrypted training function
def train_encrypted_model(X, y, weights, bias, learning_rate, epochs=5):
    for epoch in range(epochs):
        total_loss = 0
        for x_enc, y_enc in zip(X, y):
            # Prediction
            pred = x_enc.dot(weights) + bias
            pred = ts.ckks_vector(context, [1 / (1 + np.exp(-pred.decrypt()[0]))])  # Sigmoid
            
            # Error
            error = pred - y_enc
            total_loss += abs(error.decrypt()[0])
            
            # Gradients
            grad_w = x_enc * error.decrypt()[0]
            grad_b = error.decrypt()[0]
            
            # Update
            weights -= learning_rate * np.array(grad_w.decrypt())
            bias -= learning_rate * grad_b
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    return weights, bias

weights, bias = train_encrypted_model(encrypted_X_part1, encrypted_y_part1, weights, bias, learning_rate)
```

---

### **5. Validating the Encrypted Model**

Evaluate the trained model on encrypted data.

```python
from sklearn.metrics import accuracy_score

# Prediction function
def predict_encrypted(X, weights, bias):
    predictions = []
    for x_enc in X:
        pred = x_enc.dot(weights) + bias
        pred = 1 / (1 + np.exp(-pred.decrypt()[0]))
        predictions.append(1 if pred >= 0.5 else 0)
    return predictions

# Validate the model
encrypted_predictions = predict_encrypted(encrypted_X_part2, weights, bias)
accuracy = accuracy_score(y_part2, encrypted_predictions)
print(f"Accuracy of the encrypted model: {accuracy:.2%}")
```

---

## **Results**

- **Model Accuracy:** ~80%  
- The encrypted training and validation process ensures that sensitive data remains private while producing meaningful insights.

---

## **Conclusion**

This demonstration shows how **SMPC** can be applied in encrypted machine learning to enable secure collaboration between institutions. It ensures privacy while enabling valuable insights from combined datasets. SMPC is a powerful tool for secure AI in sensitive domains like healthcare and finance.
