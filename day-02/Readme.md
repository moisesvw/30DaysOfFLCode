# Exploring Federated Learning for Credit Card Fraud Detection (FedAvg)

In this exploration, I aim to demonstrate how **Federated Learning (FL)** could potentially be applied to detect credit card fraud in a privacy-preserving manner. I focus on using the **Federated Averaging (FedAvg)** technique and leverage **SyftBox** to simulate a distributed network of entities with private data. The goal here is to understand and test the feasibility of these technologies rather than to propose a production-ready solution.

---

## **Day 2: Establishing the Use Case and Exploring FL Techniques**

### **Objective**
Today, I:
1. Defined a **fraud detection use case** using the Kaggle dataset.
2. Explored how **Federated Learning** concepts could be applied to this dataset.
3. Planned an implementation of **FedAvg** to train a global model across distributed datasets.

---

## **Introduction**

### **Use Case: Fraud Detection**
Detecting fraudulent credit card transactions is a critical challenge for financial institutions. Privacy regulations often prevent these entities from sharing sensitive customer data. Federated Learning offers a theoretical framework to collaborate on building robust fraud detection models while keeping data private. This exploration seeks to simulate such a scenario to understand the potential benefits and limitations.

---

## **Dataset**

### **Initial Dataset**
For this exploration, I use a simulated dataset from Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection). It contains:
- **Legitimate and fraudulent transactions** from 1000 customers and 800 merchants.
- A time period covering **Jan 2019 - Dec 2020**.

### **Preprocessed Dataset**
To facilitate the exploration, I preprocess the dataset to:
- Remove unnecessary columns (e.g., customer names, geographic coordinates).
- Calculate the distance between customer and merchant.
- Encode categorical features to prepare them for machine learning.

Refer to [Data_Processing.md](Data_Processing.md) for detailed preprocessing steps.

---

## **Federated Learning: Key Techniques**

### **Federated Averaging (FedAvg)**
FedAvg is the main approach I plan to explore for aggregating updates from multiple models trained on distributed datasets:
1. Each **Datasite** trains a local model using its private data.
2. The trained model's weights or gradients are sent to a **leader node**.
3. The leader node computes a **weighted average** of the updates to create a global model.
4. The global model is redistributed to the Datasites for further training.

### **Why Explore FedAvg?**
- **Privacy-Preserving:** Only updates (not raw data) are shared with the leader.
- **Scalable:** Designed to handle large numbers of distributed nodes.
- **Flexible:** Can adapt to varying data distributions across nodes.

### **Future Integration with SyftBox**
In the next steps, I aim to:
1. Simulate three **Datasites** using SyftBox to represent entities with private datasets.
2. Distribute the fraud detection dataset across these Datasites.
3. Implement FedAvg to aggregate updates in the leader node and refine a global model.

---

## **Setup Instructions**

### **Install Dependencies**
To prepare for data preprocessing and model experimentation, I set up the following environment:
```bash
# Create a virtual environment
python -m venv day-02
source day-02/bin/activate

# Install required packages
pip install pandas scikit-learn geopy
```
