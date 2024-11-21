# Day 1: Exploring Federated Learning and SyftBox

Welcome to Day 1 of the **#30DaysOfFLCode challenge**! This repository documents my journey into Federated Learning (FL) and privacy-enhancing technologies (PETs) using **SyftBox**. Below, you'll find:  
1. A summary of key concepts from a foundational Federated Learning paper.  
2. An introduction to SyftBox, its features, and why it's important.  
3. Step-by-step instructions to set up SyftBox and run a CPU usage sharing demo.  

---

## Step 1: Key Concepts from the Paper

Federated Learning is a decentralized approach to training machine learning models. Instead of centralizing sensitive data, models are trained locally, and updates (e.g., weights, gradients) are aggregated into a global model. The paper highlights critical techniques:

### **1️⃣ Federated Averaging (FedAvg)**  
Combines updates from distributed models to build a global model. This technique balances efficiency and accuracy, ensuring data remains private on local devices.

### **2️⃣ Differential Privacy**  
Adds noise to model updates to protect individual data points. Differential Privacy ensures sensitive patterns in the data are not exposed, even indirectly.

### **3️⃣ Communication Efficiency**  
Techniques like sparsification and quantization reduce the amount of data exchanged between devices and servers, addressing communication bottlenecks.

### **4️⃣ Homomorphic Encryption**  
Allows computations to be performed directly on encrypted data, ensuring sensitive information remains secure even during processing.

---

## Step 2: What is SyftBox?

**SyftBox**, developed by OpenMined, is a tool that simplifies the implementation of Federated Learning and other PETs. Think of it as a "Dropbox for data and computation" that operates in a decentralized manner. Here's why it stands out:

### **Key Features:**
- **Distributed Network:** Each node, called a Datasite, contributes private data and applications to the network.
- **Language Agnostic:** APIs can be written in any programming language and executed in various environments.
- **Modular Design:** Breaks down complex systems into reusable components for machine learning, analysis, and visualization.
- **Privacy-Preserving ML:** Enables collaboration while ensuring sensitive data stays local.

---

## Step 3: Code Example - Sharing CPU Usage with SyftBox

To understand SyftBox in action, I ran a simple demo that shares CPU usage data across a distributed network. This is a starting point to explore how Datasites interact securely.

### **Installation**

1. **Install SyftBox:**
   Run the following command to install SyftBox:
   ```bash
   curl -LsSf https://syftbox.openmined.org/install.sh | sh
   ```

2. **Activate the Environment and Install Syft:**
   ```bash
   uv pip install -U syftbox --quiet
   . .venv/bin/activate
   ```

---

### **Setting Up and Running the CPU Usage Demo**

Follow these steps to set up and run the CPU usage tracker API:

1. Navigate to the SyftBox folder:
   ```bash
   cd SyftBox/
   ls
   ```

2. Go to the `apis` directory:
   ```bash
   cd apis/
   ls
   ```

3. Remove any existing `cpu_tracker` folder:
   ```bash
   rm -rf cpu_tracker/
   ```

4. Clone the `cpu_tracker_member` repository and rename it:
   ```bash
   git clone git@github.com:OpenMined/cpu_tracker_member.git
   mv cpu_tracker_member/ cpu_tracker
   ```

5. Verify the structure of the `apis` directory:
   ```bash
   ls
   ```

6. Run the API script:
   ```bash
   sh run.sh
   ```

7. After testing the script, stop the Syft client and restart it using the installation command:
   ```bash
   curl -LsSf https://syftbox.openmined.org/install.sh | sh
   ```

---

## What’s Next?

1. **Establish the Financial Fraud Use Case:**
   - Use the Kaggle credit card fraud dataset to simulate private datasets across multiple Datasites.
   - Train local models using SyftBox APIs and apply **Federated Averaging (FedAvg)**.

2. **Enhance Privacy with Differential Privacy and Homomorphic Encryption:**
   - Protect model updates by adding noise and enabling secure encrypted computations.

3. **Document and Share Progress:**
   - Track daily learnings and share them through LinkedIn and GitHub.

---

This README lays the groundwork for understanding SyftBox and Federated Learning while setting up a practical demo. Ready to explore privacy-preserving ML further? 🚀
