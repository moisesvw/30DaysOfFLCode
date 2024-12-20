# **Day 30: Exploring SyftBox Applications and Wrapping Up**

Today marks the final day of my **#30DaysOfFLCode** challenge! I want to thank the OpenMined community, Andrew Trask, and all the contributors who have made this an incredible journey into the world of **Federated Learning (FL)** and **Privacy-Enhancing Technologies (PETs)**.

To wrap up this journey, I explored two fascinating projects from the OpenMined community built on **SyftBox**, showcasing real-world applications of FL and PETs.

---

## **Application 1: SyftBox Netflix Viewing History App**

### **Overview**
The **Netflix Viewing History App** is built on **SyftBox** to process and analyze Netflix viewing histories while preserving user privacy. The app demonstrates how **Federated Learning** can provide aggregated insights into user behaviors without compromising individual data security.

#### **Key Features:**
- Aggregates popular shows across users.
- Detects trends in viewing habits while keeping data private.
- Compares individual behaviors to anonymized group patterns.

### **Setup Instructions**

#### **1. Install SyftBox**
If SyftBox is not already set up, you can follow my guide for installing it via Docker:

[Day 24: SyftBox Setup](https://github.com/moisesvw/30DaysOfFLCode/tree/main/day-24)

#### **2. Install the Netflix App**
Clone the Netflix App from the community repository and copy your Netflix history file into a private folder within your SyftBox environment:

```bash
syftbox app install gubertoli/syftbox-netflix --config /build/config.json
docker cp ~/Downloads/NetflixViewingHistory.csv driftbox:/build/data/private/NetflixViewingHistory.csv
```

#### **3. Configure the Environment**

- Navigate to the app directory:

  ```bash
  cd /build/data/apis/syftbox-netflix
  vim .env
  ```

- Update the `.env` file to set the **AGGREGATOR_DATA_DIR** to your private folder:

  ```bash
  AGGREGATOR_DATA_DIR="/build/data/private"
  ```

#### **4. Run the App**
Follow the repository’s instructions to process and analyze your Netflix data while ensuring complete privacy.

[Netflix Viewing History App Repository](https://github.com/gubertoli/syftbox-netflix)

---

## **Application 2: Federated RAG (Retrieve and Generate) App**

### **Overview**
The **Federated RAG App** showcases a fully encrypted **Retrieve and Generate (RAG)** pipeline built on **SyftBox**. This app integrates PETs like **Homomorphic Encryption** to enable private information retrieval and generation workflows.

#### **Key Use Cases:**
- Peer-matching applications.
- Code quality checks and project searches.
- Generative AI workflows for distributed datasets.

### **Setup Instructions**

#### **1. Clone the RAG Repository**

```bash
docker exec -it driftbox bash
cd /data/apis/
git clone https://github.com/siddhant230/federated_rag.git
```

#### **2. Install and Configure SyftBox**
If SyftBox is not yet installed, refer to the instructions in the [Day 24 guide](https://github.com/moisesvw/30DaysOfFLCode/tree/main/day-24) or the RAG repository’s setup guide.

#### **3. Explore the App**
Follow the steps in the [Federated RAG App Repository](https://github.com/siddhant230/federated_rag) to explore its capabilities and learn how RAG can enable privacy-focused workflows for distributed datasets.

---

### **Thank You**
A heartfelt thanks to:
- **Andrew Trask** and the **OpenMined team** for creating an amazing community.
- All the developers who contributed to the projects I explored. Your work is inspiring and drives innovation in privacy-preserving AI.
- Everyone following along and encouraging this journey. Your support means everything.

Here’s to continuing the journey of learning and building privacy-preserving AI! 🌌

