# **Day 16: Exploring the Heart Disease Dataset with PySyft's Distributed Workflow**

## **Overview**

I explored how to securely train Machine Learning models on sensitive medical datasets hosted on distributed servers using **PySyft**. Following the [Heart Disease tutorial](https://github.com/OpenMined/syft-heart-disease-tutorial), I set up multiple local Datasites to emulate a distributed, privacy-preserving environment for collaborative learning.

---

### **What was it about?**

The tutorial demonstrated how to set up a distributed environment to securely work with sensitive medical data across institutions. I used a dataset called "Heart Study Data," which was shared among multiple servers to facilitate collaborative machine learning analysis.

### **Why is it important?**

This practice is critical for learning how to manage distributed data in Machine Learning (ML) projects, especially in fields like healthcare, where data is fragmented across institutions and must remain private. It also provides insight into collaborating with decentralized data without compromising security or privacy, an essential aspect of training ML models in real-world contexts.

### **What did I learn?**

- **Managing distributed data:** I learned how to split data across multiple servers and prepare it for collaborative analysis.
- **Applications in ML:** I understood how to use such configurations to train ML models in distributed systems without centralizing the data, following privacy principles like those used in **federated learning**.
- **Uploading and managing datasets:** I learned how to load and validate medical data for use in ML analysis scenarios.
- **Privacy in medical data:** I reflected on implementing secure systems that allow the use of sensitive data without violating regulations such as HIPAA or GDPR.

## **Setup and Initialization**

### **Step 1: Clone the Repository and Install Dependencies**

To get started, I cloned the repository and installed the required dependencies.

```bash
git clone https://github.com/OpenMined/syft-heart-disease-tutorial.git
cd syft-heart-disease-tutorial
pip install -r requirements.txt
````

### **Step 2: Launch Datasites**

I launched four local Datasites to simulate distributed medical institutions:

```bash
python launch_datasites.py
```

### **Available Datasites**

1. **Cleveland Clinic**: `http://localhost:54879`
2. **Hungarian Inst. of Cardiology**: `http://localhost:54880`
3. **Univ. Hospitals Zurich and Basel**: `http://localhost:54881`
4. **V.A. Medical Center**: `http://localhost:54882`

---

## **Key Steps in Training on Distributed Datasites**

### **Step 1: Login to Datasites**

Securely login to all Datasites to access their hosted datasets.

```python
import syft as sy

DATASITE_URLS = {
    "Cleveland Clinic": "http://localhost:54879",
    "Hungarian Inst. of Cardiology": "http://localhost:54880",
    "Univ. Hospitals Zurich and Basel": "http://localhost:54881",
    "V.A. Medical Center": "http://localhost:54882",
}

datasites = {}
for name, url in DATASITE_URLS.items():
    datasites[name] = sy.login(url=url, email="researcher@openmined.org", password="****")

print(f"Logged into {len(datasites)} Datasites.")
```

### **Step 2: Validate Training Pipeline with Mock Data**

Using mock data ensures that the training pipeline works without exposing sensitive information.

#### **Retrieve Mock Data**

```python
mock_data = datasites["Cleveland Clinic"].datasets["Heart Disease Dataset"].assets["Heart Study Data"].mock
```

#### **Data preparation and submit request code execution**
- **Dataset Preparation**:
    - Connects to a distributed dataset labeled "Heart Disease Dataset" hosted at various "datasites."
    - Accesses mock data for processing and analysis.
- **Data Aggregation**:
    - Maps the age of individuals into three ranges: "0-40," "40-65," and "Over 65."
    - Converts diagnosis results into binary categories ("present" or "absent") for clarity.
    - Decodes the "sex" column into human-readable labels ("male" and "female").
- **Disease Prevalence Calculation**:
    - Uses demographic factors (age range, sex, and diagnosis) to calculate the prevalence of heart disease within groups.
    - Generates a contingency table showing the distribution of diagnosis across combinations of age range and sex.
- **Distributed Execution**:
```python
    @sy.syft_function_single_use(data=data_asset)
    def disease_prevalence_per_demographic(data) -> pd.DataFrame:
    # .
    # .
    # .
    datasite.code.request_code_execution(disease_prevalence_per_demographic)
```

    - Defines a secure function (`sy.syft_function_single_use`) to calculate disease prevalence on data residing on remote servers, preserving data privacy.
    - Requests the remote execution of this function across all datasites hosting the dataset.

#### **Train a Random Forest Model**

This is an excerpt from the notebook tutorial, where a function `train` is defined and decorated with `sy.syft_function_single_use`. For each Datasite, this function is customized and submitted to the respective server. The function uses real data on the Datasite for training the model, and the results are collected in subsequent steps.

```python
for name, datasite in datasites.items():
    print(f"Datasite: {name}")
    # 1. Get data asset from datasite
    data_asset = datasite.datasets["Heart Disease Dataset"].assets["Heart Study Data"]
    
    @sy.syft_function_single_use(data=data_asset)
    def train(data):
	    # .
	    # .
	    # . OTHER CODE
	    # .
        # 3. train model
        model = RandomForestClassifier(random_state=12345)
        model.fit(X_train, y_train)
        # 4. model persistance - return model serialised 
        serialised_model = BytesIO()
        joblib.dump(model, serialised_model)

        return serialised_model
    
    ml_training_project = sy.Project(
        name="Traning RandomForest Classifier on Heart Study Data",
        description="""I would like to train a classifier on the Heart Study data.
        The code will partition the dataset using sex and target, and will train 
        a RandomForest classifier, that will be returned serialised.
        """,
        members=[datasite],
    )
    ml_training_project.create_code_request(train, datasite)
    project = ml_training_project.send()
```


---

## **Key Learnings and Insights**

### **1. Privacy-Preserving Machine Learning**

This tutorial highlighted how sensitive datasets can remain private during collaborative model training. By keeping data local to Datasites and running experiments remotely, I ensured compliance with privacy regulations like GDPR and HIPAA.

### **2. Distributed Learning in Healthcare**

Simulating multiple medical institutions demonstrated how Federated Learning enables collaboration without centralizing sensitive data.

### **3. Validating Training Pipelines**

Testing with mock data helped to identify potential issues in the training code before running experiments on actual data.

---

## **Next Steps**

In the next steps, I plan to:

**Experiment with ensemble methods**: Combine models trained on different Datasites for improved performance.
