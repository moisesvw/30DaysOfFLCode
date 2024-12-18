# **Day 28: Exploring "Vertical Federated Learning without Revealing Intersection Membership"**

Today, I dove into the paper:  
[**"Vertical Federated Learning without Revealing Intersection Membership"**](https://arxiv.org/pdf/2106.05508).

This paper explores how to mitigate privacy risks in **Vertical Federated Learning (vFL)** by addressing **intersection membership leakage** during data alignment. It introduces innovative techniques to ensure privacy while maintaining high model utility, paving the way for broader applications of vFL in privacy-sensitive industries like healthcare and finance.

---

## **Key Insights from the Paper**

### **1. Problem Addressed**

In traditional vFL setups, parties use **Private Set Intersection (PSI)** to identify overlapping data entities. However, PSI leaks information about which entities are shared between parties, leading to **intersection membership leakage** and privacy concerns.

### **2. Significance**

- **Privacy Risks**: Sharing intersection membership can violate regulations like GDPR and HIPAA.
- **Necessity**: A privacy-preserving method for aligning data in vFL is essential for secure collaboration among organizations.

### **3. Techniques Introduced**

The paper proposes a **Private Set Union (PSU)** protocol that avoids revealing intersection membership and combines it with synthetic data generation for non-intersecting samples. It also introduces **logits calibration** to maintain model utility when using synthetic data.

---

## **Core Techniques**

### **1. Private Set Union (PSU)**

- Based on **Diffie–Hellman** cryptographic principles.
- Outputs the union of datasets without revealing which entities are shared.
- Ensures that data entities remain private even during alignment.

### **2. Synthetic Data Generation**

- For non-intersecting samples, synthetic data is generated to maintain alignment.
- Synthetic labels are derived using strategies like majority voting, and synthetic features are sampled from approximated distributions of the real data.

### **3. Logits Calibration**

- Introduced to adjust the model’s logits when trained on mixed real and synthetic data.
- Reduces errors caused by synthetic data, improving the reliability of predictions.

---

## **Applications**

The techniques presented are crucial for industries where privacy and compliance are non-negotiable:

1. **Healthcare**: Securely train models on patient records across multiple institutions.
2. **Finance**: Collaborate on fraud detection models without sharing sensitive client information.
3. **Marketing**: Enable cross-organization analysis without disclosing customer identities.

---

## **Code Implementation**

To solidify my understanding, I implemented parts of the techniques described in the paper.

### **Private Set Union Example**

```python
import random

# Simulate datasets from two parties
# Two parties (A and B) each have a set of IDs representing their data.
party_A_data = {1, 2, 3, 4}
party_B_data = {3, 4, 5, 6}

# Prime number and generator for Diffie-Hellman
# p is a "safe prime", which is a prime number where (p - 1) / 2 is also prime.
# g is a "generator", a number that generates a large subset of unique results modulo p
# when raised to all powers from 1 to p - 1. It is critical for Diffie-Hellman key exchange.
p = 23  # Safe prime
g = 5   # Generator

# Generate random secrets
# Each party generates a random secret. This secret acts as their private key for the protocol.
# These secrets should be kept private and not shared with the other party.
a_secret = random.randint(1, p - 1)  # Party A's secret
b_secret = random.randint(1, p - 1)  # Party B's secret

# Hash data from both parties
# Each party "hashes" its data by raising the generator g to the power of the product of their secret
# and each data point, modulo p. This creates a unique, secure representation of their data.
hashed_A = {pow(g, x * a_secret, p) for x in party_A_data}
hashed_B = {pow(g, x * b_secret, p) for x in party_B_data}

# Exchange hashes and rehash
# Each party receives the hashed data from the other party and rehashes it using their secret.
# This step ensures that both parties can securely compute shared representations of the data
# without revealing their original data points or secrets.
rehash_A = {pow(y, a_secret, p) for y in hashed_B}  # Party A rehashes data from Party B
rehash_B = {pow(y, b_secret, p) for y in hashed_A}  # Party B rehashes data from Party A

# Compute the union
# The union of the rehashed sets represents all unique elements from both parties,
# but in their hashed form, maintaining privacy.
union_hashed_ids = rehash_A.union(rehash_B)
print("Union of IDs (hashed):", union_hashed_ids)

```

### **Synthetic Data Generation Example**

```python
import numpy as np

# Example feature distribution
real_features = np.random.normal(loc=0, scale=1, size=(100, 10))

# Generate synthetic features
def generate_synthetic_features(real_data, size):
    mean = np.mean(real_data, axis=0)
    std = np.std(real_data, axis=0)
    return np.random.normal(loc=mean, scale=std, size=(size, real_data.shape[1]))

synthetic_features = generate_synthetic_features(real_features, 20)
print("Synthetic Features:\n", synthetic_features)
```

---

## **Impact and Key Takeaways**

1. **Privacy Preservation**:
    
    - PSU avoids leaking intersection membership.
    - Synthetic data ensures privacy for non-intersecting samples.
2. **Model Utility**:
    
    - Despite using synthetic data, the model maintains competitive performance, with minimal AUC drops (≤5%).
3. **Scalability**:
    
    - The cryptographic techniques scale well with dataset size, ensuring usability for large datasets.
4. **Regulatory Compliance**:
    
    - This approach adheres to privacy laws like GDPR and HIPAA, making it suitable for sensitive industries.

---

This paper presents a significant advancement in privacy-preserving techniques for Vertical Federated Learning. The proposed **PSU protocol** and **synthetic data generation** offer practical solutions to address intersection membership leakage, enabling secure collaborations across organizations. This exploration deepened my understanding of privacy in federated systems and inspired new ideas for applying these techniques in real-world scenarios.
