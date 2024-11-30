# **Day 9: Reviewing and Executing Federated Learning Experiments in PySyft**

Today, I continued exploring **PySyft** and focused on completing the Federated Learning (FL) workflow. Specifically, I worked as a data owner to review, approve, and execute code requests submitted by external data scientists. This crucial step ensures that experiments run securely on real data without exposing sensitive information, maintaining privacy throughout the collaboration.

---

## **Objectives**

1. Understand how to review and manage code requests as a data owner.
2. Approve or reject proposed experiments based on security and privacy criteria.
3. Execute approved experiments and analyze the results generated from real data.
4. Address challenges encountered during the Federated Learning process.

---

## **Setup**

To build upon the previous workflow (Day 8), I ensured the following:

- The **Datasite** server was running and accessible.
- The experiment submitted by the data scientist was ready for review.

---

## **Workflow**

### **1. Log in as Data Owner**

To begin, I logged into the PySyft client as the data owner, gaining access to the submitted code requests for review.

```python
import syft as sy

# Login as the data owner
client = sy.login(url="localhost:8081", email="info@openmined.org", password="changethis")
```

### **2. Review Code Requests**

The submitted code requests were listed and inspected to ensure compliance with privacy and security standards.

```python
# List all projects and their code requests
projects = client.projects
for project in projects:
    print(f"Project: {project.name}")
    code_requests = client.requests
    for request in code_requests:
        print(f"Code Request ID: {request.id}, Status: {request.status}")
        print(f"Description: {request}")

# Select a specific request for review
request = client.requests[0]
request
```

### **3. Test the Submitted Code**

To validate the request, I executed the experiment on **mock data** to identify any issues before running it on the **real data**.

```python
# Access the submitted code
syft_function = request.code

# Load the dataset and assets
fraud_dataset = client.datasets["Fraud Detection Dataset"]
features, labels = fraud_dataset.assets

# Test the experiment on mock data
result_mock_data = syft_function.run(features_data=features.mock, labels=labels.mock)
print("Mock Data Result:", result_mock_data)
```

### **4. Address Issues**

During testing, I encountered a coding error where an undefined variable `X` was referenced. The request was denied with feedback for the data scientist to submit a corrected version.

```python
request.deny(reason="Code contains undefined variables.")
```

After the corrected code was submitted, I retested the experiment on mock data, approved the request, and ran it on the real data:

```python
# Run the corrected experiment on mock data
result_mock_data = syft_function.run(features_data=features.mock, labels=labels.mock)
print("Mock Data Result:", result_mock_data)

# Run the experiment on real data
result_real_data = syft_function.run(features_data=features.data, labels=labels.data)
print("Real Data Result:", result_real_data)

# Approve the request
request.approve()
```

### **5. Analyze Results**

In the role of a data scientist, I reviewed the submitted requests and noticed some were denied. Upon receiving the first approval, I attempted to execute the experiment to retrieve the results. However, the experiment failed as the PySyft server did not support the XGBoost library. To address this, I updated the code to use a Random Forest Classifier instead. Below is the revised implementation.

```python
result = client.code.ml_experiment_on_fraud_data(features_data=features, labels=labels).get()
print(f"Training Accuracy: {result[0]}, Test Accuracy: {result[1]}")
```

Example results:

- Training Accuracy: **99.89%**
- Test Accuracy: **99.61%**

### **Updated Model Code**

The updated Random Forest Classifier replaced the initial unsupported XGBoost model:

```python
def ml_experiment_on_fraud_data(features_data, labels, seed: int = 12345) -> tuple[float, float]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.model_selection import train_test_split

    # Train the model using RandomForestClassifier
    def train_random_forest(X_train, y_train):
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model

    # Evaluate the model
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        print("Classification Report:", classification_report(y_test, y_pred))
        return accuracy_score(y_test, y_pred)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features_data, labels, test_size=0.25, random_state=seed, stratify=labels
    )

    # Train and evaluate the model
    model = train_random_forest(X_train, y_train)
    return evaluate_model(model, X_train, y_train), evaluate_model(model, X_test, y_test)
```

---

## **Insights**

### Challenges Encountered:

1. **Undefined Variables**: The submitted code had missing definitions, requiring resubmission.
2. **Library Restrictions**: XGBoost was not supported in the Datasite server, prompting a switch to Random Forest.
3. **Docker Networking**: Adjustments to the Docker environment were needed to resolve hostname issues.

### Key Takeaways:

- PySyft provides a robust framework for ensuring data privacy while enabling meaningful collaboration.
- Data scientists can run experiments on sensitive data without direct access, ensuring privacy for data owners.
- The iterative process of reviewing and approving requests helps maintain high standards for security and functionality.

---

## **Next Steps**

1. Summarize learnings and explore additional privacy-preserving techniques in Federated Learning.
2. Extend this workflow to include more complex use cases.
3. Share insights gained with the broader community to demonstrate PySyft’s potential in real-world applications.

---

This day’s work highlights the elegance and power of PySyft in fostering secure, privacy-preserving machine learning collaborations.
