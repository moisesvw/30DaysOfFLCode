# **Day 15: Exploring Federated Reconstruction (FedRecon)**

## **What is Federated Reconstruction (FedRecon)?**

**Federated Reconstruction (FedRecon)** is an innovative approach in **Federated Learning (FL)** that addresses the challenges of large-scale distributed training. FedRecon introduces a hybrid parameter framework combining:

- **Global Parameters (G):** Shared and updated across all clients.
- **Local Parameters (l_i):** Specific to each client, reconstructed on demand without being stored persistently.

---

## **Problems Solved by FedRecon**

1. **Stateless Clients:**
   - Unlike traditional methods, FedRecon avoids storing local parameters on devices, making it scalable to billions of clients.
2. **Support for Unseen Users:**
   - Clients who didn’t participate in training can still perform inference by reconstructing their local parameters based on the shared global model.
3. **Heterogeneous Data:**
   - Captures user-specific patterns through localized training, improving performance in non-IID data scenarios.

---

## **How FedRecon Works**

### **Training Workflow**

1. **Global Parameter Sharing:**
   - The central server sends **global parameters (G)** to each participating client.
2. **Local Parameter Reconstruction:**
   - Each client freezes **G** and reconstructs **local parameters (l_i)** using their private data.
3. **Global Updates:**
   - Clients freeze **l_i** and compute updates for **G** based on local data.
4. **Server Aggregation:**
   - The central server averages the updates to produce the new global parameters **G'**.

---

## **Practical Example: Federated Reconstruction for Matrix Factorization**

### **Problem Setup**

We simulate a **matrix factorization problem** commonly used in recommendation systems. Given a user-item rating matrix, we aim to predict missing ratings while keeping user-specific data private.

---

### **Step 1: Simulate Distributed Data**

Each client has access to a row of the rating matrix.

```python
import numpy as np

# Simulate a user-item rating matrix
R = np.array([
    [5, 0, 3, 0],
    [4, 0, 0, 2],
    [0, 1, 0, 4],
    [0, 0, 5, 0],
])

print("Rating Matrix:")
print(R)

# Partition data by users
user_data = {f"user_{i}": R[i, :] for i in range(R.shape[0])}
```

---

### **Step 2: Initialize Global and Local Parameters**

- **Global Parameters (G):** Represent item embeddings.
- **Local Parameters (l_i):** Represent user embeddings, reconstructed during each round.

```python
# Initialize global (G) and local (l_i) parameters
num_items = R.shape[1]
embedding_dim = 2

G = np.random.rand(num_items, embedding_dim)  # Global item embeddings
```

---

### **Step 3: Local Reconstruction**

Each client reconstructs its **local parameters (l_i)** using its private data and the shared **global parameters (G)**.

```python
def reconstruct_local(user_ratings, G, learning_rate=0.01, epochs=10):
    l_i = np.random.rand(embedding_dim)  # Initialize local parameters
    for _ in range(epochs):
        for item_idx, rating in enumerate(user_ratings):
            if rating > 0:  # Train only on known ratings
                error = rating - np.dot(l_i, G[item_idx])
                l_i += learning_rate * error * G[item_idx]  # Update local parameters
    return l_i

# Local reconstruction for a single client
user_0_ratings = user_data["user_0"]
l_0 = reconstruct_local(user_0_ratings, G)
print("\nReconstructed Local Parameters for User 0:", l_0)
```

---

### **Step 4: Global Update**

Clients compute gradients for the **global parameters (G)** while freezing their **local parameters (l_i)**.

```python
def update_global(user_ratings, G, l_i, learning_rate=0.01):
    grad_G = np.zeros_like(G)
    counts = np.zeros(G.shape[0])  # Count updates for each item

    for item_idx, rating in enumerate(user_ratings):
        if rating > 0:
            error = rating - np.dot(l_i, G[item_idx])
            grad_G[item_idx] += learning_rate * error * l_i
            counts[item_idx] += 1

    # Average gradients for items with updates
    for idx in range(len(grad_G)):
        if counts[idx] > 0:
            G[idx] += grad_G[idx] / counts[idx]

    return G

# Perform global update for a single client
G = update_global(user_0_ratings, G, l_0)
print("\nUpdated Global Parameters (G):", G)
```

---

## **Reflection on FedRecon**

### **Why Choose FedRecon Over FedAvg?**

1. **Different Goals:**
   - **FedAvg** averages entire models, ideal for homogeneous data.
   - **FedRecon** separates global and local parameters, excelling in heterogeneous data scenarios.
2. **Generalization:**
   - FedRecon enables unseen clients to participate by reconstructing their local parameters.

---

### **Key Properties of FedRecon**

1. **Stateless Clients:**
   - No need to store parameters between rounds, simplifying training at scale.
2. **Support for Unseen Clients:**
   - Unseen users can leverage the global model by reconstructing their local parameters.
3. **Privacy Preservation:**
   - Local parameters and gradients remain private, protecting sensitive user data.

---

## **Next Steps**

1. **Experiment with Larger Matrices:**
   - Scale the example to include more users and items.
2. **Evaluate Performance Metrics:**
   - Analyze accuracy and convergence speed of FedRecon compared to FedAvg.
3. **Explore Practical Use Cases:**
   - Implement FedRecon in real-world scenarios like personalized recommendations or collaborative filtering.


## **Acknowledgment**

The principles and implementation details of Federated Reconstruction were adapted from Google's research blog:
"A Scalable Approach for Partially Local Federated Learning".

https://research.google/blog/a-scalable-approach-for-partially-local-federated-learning/

