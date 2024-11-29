# Day 8: Running ML Experiments with Mock Data on PySyft Datasite

Today, I explored the workflow of a **Data Scientist** in a Federated Learning setup using PySyft. Building on the environment and Datasite created in previous days, I simulated the role of a Data Scientist connecting to the Datasite. The key goal was to learn how to access mock data, prepare an ML experiment, and propose the execution of the experiment using real data—without ever directly accessing sensitive information.

---

## **Key Accomplishments**

1. **Emulated the Data Scientist Role**:
   - Created a user with restricted access to only mock data.
   - Connected to the Datasite as a Data Scientist to list available datasets and load mock data.

2. **Prepared an ML Experiment**:
   - Used mock data to create a pipeline for predicting fraudulent transactions.
   - Developed an XGBoost model for experimentation.

3. **Proposed an Experiment**:
   - Submitted the experiment as a project request using PySyft’s `syft_function_single_use` decorator to ensure secure code execution.

To accomplish this task, I followed the official PySyft tutorial on running a research study, which provided step-by-step guidance. You can find the tutorial here: PySyft Research Study Guide.

https://docs.openmined.org/en/latest/getting-started/part3-research-study.html

---

## **Why This is Powerful**

This pipeline showcases how PySyft empowers secure collaboration in Federated Learning by:
- Allowing Data Scientists to work on **mock data** for experimentation.
- Enabling the **Datasite Owner** to approve or deny experiments before execution.
- Ensuring that sensitive data remains private while supporting meaningful analysis.

By simulating a real-world scenario, this framework allows organizations to unlock the potential of their data while respecting privacy constraints.

---

## **Workflow Steps**

### **1. Creating the Data Scientist User**

The Datasite Owner creates a user for the Data Scientist, granting limited access to mock data:

```python
import syft as sy

client = sy.login(url="localhost:8081", email="info@openmined.org", password="changethis")

datascientist_account_info = client.users.create(
    email="datascientist@datascience.inst",
    name="Data Scientist",
    password="syftrocks",
    password_verify="syftrocks",
    institution="Data Science Institute",
    website="https://datascience_institute.research.data"
)
```

### **2. Accessing the Datasite as a Data Scientist**

The Data Scientist logs in, lists datasets, and loads mock data for exploration:

```python
import syft as sy

client = sy.login(url="localhost:8081", email="datascientist@datascience.inst", password="syftrocks") 

# List datasets
fraud_dataset = client.datasets["Fraud Detection Dataset"]

# Load features and targets
features, targets = fraud_dataset.assets

# Accessing real data raises an error
# features.data  # Syft Permission Error

# Explore mock data
print(features.mock.head())
print(targets.mock.head())
```

---

### **3. Preparing the ML Experiment**

Using mock data, the Data Scientist creates a pipeline to train and evaluate an XGBoost model:

```python
!pip install scikit-learn
!pip install xgboost
```

```python
def ml_experiment_on_fraud_data(features_data, labels, seed: int = 12345) -> tuple[float, float]:
    from sklearn.metrics import classification_report, accuracy_score
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features_data, labels, test_size=0.25, random_state=seed, stratify=labels)
    
    # Train XGBoost model
    model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
```

Run the experiment locally on mock data to ensure it works as expected:

```python
ml_experiment_on_fraud_data(features_data=features.mock, labels=targets.mock)
```

---

### **4. Proposing the Experiment**

To submit the experiment for approval, the code is wrapped using `syft_function_single_use` and included in a project proposal:

```python
remote_user_code = sy.syft_function_single_use(features_data=features, labels=targets)(ml_experiment_on_fraud_data)

description = """
    The purpose of this study is to run a machine learning 
    experimental pipeline on a fraud detection dataset. 
    The initial pipeline includes preprocessing steps such as encoding 
    categorical features and normalizing numerical features. 
    The selected ML model is XGBoost, aiming to evaluate 
    accuracy scores on both training and testing data partitions 
    that are randomly generated from the mock dataset.
"""

# Create a project
research_project = client.create_project(
    name="Fraud Detection ML Project",
    description=description,
    user_email_address="analyst@datascience.org"
)

# Submit the code request
code_request = research_project.create_code_request(remote_user_code, client)
print(code_request)
```

---

## **Next Steps**

1. As the Datasite Owner, review the submitted code request.
2. Approve or deny the request based on security and feasibility.
3. If approved, execute the experiment on real data and retrieve the results.
4. Evaluate the effectiveness of the Federated Learning setup.

