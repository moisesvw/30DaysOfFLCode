# **Day 18: Federated Learning with PySyft**

## **Overview**

Today, I explored how to set up a Federated Learning (FL) system using **PySyft** and **SyftBox**, focusing on training a distributed Machine Learning model on the MNIST dataset. The process involved configuring clients, setting up an aggregator, and managing the end-to-end workflow of a Federated Learning experiment. 

This builds on my goal to:
- Implement client-side model training securely.
- Create an aggregation server to combine model updates.
- Evaluate the global model's performance.
- Understand privacy considerations in Federated Learning.

---

### **What is Federated Learning?**

**Federated Learning (FL)** enables training Machine Learning models across decentralized devices or data centers without sharing raw data. Instead, only model updates (gradients) are exchanged, ensuring data privacy while improving the global model.

Key components of FL:
1. **Clients**: Perform local training on private data and share model updates.
2. **Aggregator**: Collects updates from clients, combines them, and creates a new global model.

---

### **Step-by-Step Workflow**

#### **1. Setting Up SyftBox**

I installed SyftBox using the following command, modifying it to prevent the client from auto-starting (`ASK_RUN_CLIENT=0`):

```sh
curl -LsSf https://syftbox.openmined.org/install.sh | sed 's/ASK_RUN_CLIENT=1/ASK_RUN_CLIENT=0/' | sh
```

Next, I created an `.env` file to store the environment variables:

```sh
EMAIL=myemail@gmail.com
DATA_DIR=/build/data
CONFIG_PATH=/build/config.json
PORT=8080
```

Finally, I initialized the SyftBox client:

```sh
. ~/.zshrc && syftbox client --email $EMAIL --data-dir $DATA_DIR --config_path $CONFIG_PATH --port $PORT
```

This setup launched the client locally and created a configuration file (`config.json`) in the specified directory, storing credentials and configurations for SyftBox.

---

#### **2. Cloning the FL Repositories**

I cloned the necessary repositories for the **FL client** and **aggregator**:

```sh
# Clone and copy the client code
git clone https://github.com/openmined/fl_client.git
cp -r fl_client /build/data/apis

# Copy sample MNIST data to the private directory
cp apis/fl_client/mnist_samples/* private/fl_client/.

# Clone and copy the aggregator code
git clone https://github.com/openmined/fl_aggregator.git
cp -r fl_aggregator /build/data/apis
```

---

#### **3. Configuring the Federated Learning Experiment**

The next step was to define the roles and parameters for the experiment:
1. **Aggregator**: Manages the experiment, shares the model architecture, and provides initial weights.
2. **Clients**: Participate in training using their private datasets.

I created a configuration file (`fl_config.json`) with the following parameters:

```json
{
    "project_name": "MNIST_FL",
    "aggregator": "myemail@gmail.com",
    "participants": ["myemail@gmail.com"],
    "rounds": 3,
    "model_arch": "model.py",
    "model_weight": "global_model_weight.pt",
    "epoch": 10,
    "learning_rate": 0.1
}
```

This file specifies:
- The number of training rounds.
- Model architecture and initial weights.
- Learning rate and epochs for training.

---

#### **4. Preparing the Experiment**

I copied the required files to the aggregator’s directories:
1. **FL configuration file** (`fl_config.json`).
2. **Model architecture** (`model.py`).
3. **Initial global model weights** (`global_model_weight.pt`).

```sh
cp fl_config.json /build/data/datasites/myemail@gmail.com/api_data/fl_aggregator/launch/
cp global_model_weight.pt /build/data/datasites/myemail@gmail.com/api_data/fl_aggregator/launch/
cp model.py /build/data/datasites/myemail@gmail.com/api_data/fl_aggregator/launch/

# Copy the dataset for evaluation
cp mnist_test_data /build/data/datasites/myemail@gmail.com/private/fl_aggregator/.
```

---

#### **5. Starting the Experiment**

Once everything was set up, I initiated the experiment by moving the request folder into the running directory:

```sh
cp -r /build/data/datasites/myemail@gmail.com/api_data/fl_client/request/MNIST_FL \
      /build/data/datasites/myemail@gmail.com/api_data/fl_client/running/
```

The aggregator coordinated the experiment, and the clients processed the requests using their private data, sending model updates back to the aggregator.


---

### **Key Learnings**

1. **Federated Learning Workflow**:
   - I gained hands-on experience building and running a complete FL system.
   - Learned how to securely connect clients and aggregators using SyftBox.

2. **Privacy by Design**:
   - FL ensures sensitive data remains on client devices, only sharing updates with the aggregator.
