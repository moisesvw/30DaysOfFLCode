# **Day 17: Enhancing Predictions with Ensemble Learning on Distributed Heart Disease Data**

## **Overview**

Today, I delved into **ensemble learning** to enhance the performance of Machine Learning models trained on sensitive medical data distributed across multiple Datasites. This builds upon my ongoing work with the Heart Disease Dataset using PySyft. Following the tutorial from [OpenMined](https://github.com/OpenMined/syft-heart-disease-tutorial/blob/main/04-Ensemble-learning-Experiment.ipynb), I trained **Random Forest** classifiers individually on each Datasite and then combined their predictions using a **Voting Classifier**, showcasing the effectiveness of ensemble methods in distributed Machine Learning.

---

### **What is Ensemble Learning?**

Ensemble learning combines predictions from multiple models to achieve better performance than any single model. It is especially effective when dealing with heterogeneous or distributed data. 

#### **Common Ensemble Techniques**
1. **Bagging**: Distributes a dataset across different models and consolidates predictions.
2. **Boosting**: Sequentially trains models, where each one focuses on correcting the errors of the previous.
3. **Stacking**: Combines predictions from multiple models using a final model to produce the result.

---

### **Step 1: Baseline Results**

Initially, I trained a **Random Forest classifier** on each Datasite independently. The Matthews Correlation Coefficient (MCC) scores for individual models were:

```python
'Cleveland Clinic': 0.766
'Hungarian Inst. of Cardiology': 0.46
'Univ. Hospitals Zurich and Basel': 0.0
'V.A. Medical Center': 0.258
```

These scores highlight performance variability across Datasites due to differences in data distribution.

---

### **Step 2: Using a Voting Classifier for Ensemble Learning**

#### **Why Voting Classifier?**

A Voting Classifier consolidates predictions from multiple models. It assigns weights to each model's predictions, allowing models with better performance to have a greater influence on the final result.

#### **Implementation**

```python
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder

# Ensemble with Voting Classifier
voting_model = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()], 
    weights=[2, 1, 0.2, 0.5]  # Assign weights based on performance
)

# Workaround to directly use pre-trained models
voting_model.estimators_ = list(models.values())
voting_model.le_ = LabelEncoder().fit(y)
voting_model.classes_ = voting_model.le_.classes_
```

- **`weights`**: Reflect the performance of individual models (e.g., MCC scores). Higher weights mean more influence.
- **Soft Voting**: Models predict probabilities, and the final prediction is based on weighted averages.

---

### **Step 3: Improved Results with Ensemble Learning**

After deploying the Voting Classifier across Datasites, the MCC scores improved:

```python
'Cleveland Clinic': 0.766
'Hungarian Inst. of Cardiology': 0.583
'Univ. Hospitals Zurich and Basel': 0.330
'V.A. Medical Center': 0.308
```

This demonstrates the robustness of ensemble methods in consolidating predictions from distributed models.

---

### **Step 4: Submitting and Evaluating the Ensemble Model**

#### **Code Explanation**

1. **Serialize and Upload Model**:
    - The **Voting Classifier** is serialized and uploaded to each Datasite for evaluation.

2. **Evaluation Function**:
    - Computes metrics like **MCC** and **Confusion Matrix** for the ensemble model.

3. **Submit Code to Datasites**:
    - The evaluation code is securely sent to each Datasite for execution.

```python
from utils import serialize_and_upload

for name, datasite in datasites.items():
    print(f"Datasite: {name}")
    
    # Upload the Voting Classifier
    remote_voting_model = serialize_and_upload(model=voting_model, to=datasite)

    @sy.syft_function_single_use(data=data_asset, model=remote_voting_model)
    def evaluate(data, model):
        X = data.drop(columns=["age", "sex", "num"], axis=1)
        y = data["num"].map(lambda v: 0 if v == 0 else 1)
        
        _, X_test, _, y_test = train_test_split(
            X, y, random_state=12345, stratify=by_demographics(data)
        )
        
        classifier = joblib.load(model)
        y_pred = classifier.predict(X_test)
        return mcc_score(y_test, y_pred), confusion_matrix(y_test, y_pred)

    project = sy.Project(
        name="Evaluate Ensemble Voting Classifier on Heart Study Data",
        description="Evaluate the performance of an ensemble model combining pre-trained classifiers.",
        members=[datasite],
    )
    project.create_code_request(evaluate, datasite)
    project.send()
```

---

### **Key Takeaways**

1. **Ensemble Learning Enhances Distributed ML**:
    - Combining predictions from multiple models mitigates weaknesses and improves overall performance.

2. **Soft Voting with Weighted Influence**:
    - Assigning weights based on individual model performance ensures the ensemble is robust and optimized.

3. **Privacy-Preserving Collaboration**:
    - By running evaluation code remotely on Datasites, the raw data never leaves its source, maintaining compliance with data privacy regulations.
