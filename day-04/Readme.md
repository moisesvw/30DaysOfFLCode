# Exploring Global and Local Differential Privacy

I aim to expand on the fundamental concepts behind Differential Privacy (DP), including Global and Local DP, Sensitivity, and noise mechanisms such as Gaussian and Laplacian distributions, along with their parameters \(\epsilon\) and \(\delta\). To proceed, I will use data from the Kaggle Fraud Detection dataset. You can find the preprocessing steps documented in the file [Data Processing](DataProcessing.md). The dataset was processed there without applying DP, so here I will focus on applying these concepts specifically to the `amt` column.


### **Concepts to Learn**

1. **Local Differential Privacy (LDP):**
   - Adds noise directly to each individual transaction amount (`amt`) before sharing.
   - Ensures that even if the raw data is leaked, individual transaction values are protected.
   - Commonly used when the data collector cannot be fully trusted.

2. **Global Differential Privacy (GDP):**
   - Adds noise to aggregated statistics derived from the data (e.g., average transaction amount).
   - The raw data is not directly perturbed; instead, noise is introduced during analysis.
   - Used when the data collector is trusted but needs to protect the results.

---

### **Step-by-Step Approach**

#### **Step 1: Analyze the Sensitivity of `amt`**
- Sensitivity measures how much a single transaction can change the result of a query (e.g., the average or sum).
- For `amt`:
  - **LDP Sensitivity:** The range of possible values (`max(amt) - min(amt)`).
  - **GDP Sensitivity for Mean:** Sensitivity divided by the number of records.

#### **Step 2: Implement Local DP**
- Perturb each individual `amt` value using Laplace noise.
- Learn how adding noise affects data utility, such as mean and variance.

#### **Step 3: Implement Global DP**
- Compute an aggregate statistic (e.g., mean or sum) and add Laplace noise.
- Compare the original and noisy aggregates.

---

### **Code Implementation**

#### **1. Analyze the Data**
```python
import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv("small_set_processed_data.csv")

# Analyze the range (sensitivity for LDP)
amt_min = data['amt'].min()
amt_max = data['amt'].max()
sensitivity_ldp = amt_max - amt_min
print(f"Range of amt (Sensitivity for LDP): {sensitivity_ldp}")

# Compute sensitivity for GDP (e.g., mean query)
n = len(data)
sensitivity_gdp_mean = sensitivity_ldp / n
print(f"Sensitivity for GDP (mean query): {sensitivity_gdp_mean}")
```

---

#### **2. Apply Local DP**
```python
# Apply Local DP using Laplace noise
epsilon_ldp = 1.0  # Privacy budget for LDP

def add_laplace_noise_ldp(value, sensitivity, epsilon):
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return value + noise

data['amt_ldp'] = data['amt'].apply(lambda x: add_laplace_noise_ldp(x, sensitivity_ldp, epsilon_ldp))

# Compare original and noisy data
print("Original amt statistics:")
print(data['amt'].describe())

print("Noisy amt (LDP) statistics:")
print(data['amt_ldp'].describe())
```

---

#### **3. Apply Global DP**
```python
# Apply Global DP to compute noisy mean
epsilon_gdp = 1.0  # Privacy budget for GDP

def add_laplace_noise_gdp(value, sensitivity, epsilon):
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return value + noise

# Compute the noisy mean
original_mean = data['amt'].mean()
noisy_mean = add_laplace_noise_gdp(original_mean, sensitivity_gdp_mean, epsilon_gdp)

print(f"Original mean of amt: {original_mean}")
print(f"Noisy mean of amt (GDP): {noisy_mean}")
```
