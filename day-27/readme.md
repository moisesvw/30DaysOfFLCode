# **Day 27: Privacy and Security in AI Systems: Threats, Risks, and Solutions**

## **Overview**

As AI-enabled systems transform the way data is managed and processed, they also introduce new vulnerabilities that threaten the privacy and security of sensitive information. Today, I explored insights from the paper [**"Privacy and Security in AI Systems"**](https://arxiv.org/html/2404.03509v1) to understand the key challenges, risks, and emerging solutions for protecting AI workflows.

---

## **1. Threats in AI Systems**

### **Internal Threats (Insiders)**

Internal actors can pose significant risks to AI systems.  
- **Key statistics**:
    - **55%** of cybersecurity incidents originate from insiders.
        - **57%** are intentional attacks by malicious employees.
        - **43%** are accidental incidents caused by negligence.
- **Example**: ChatGPT processes plaintext queries to respond, which raises confidentiality risks. Companies like Samsung have banned its use after sensitive data was unintentionally leaked by employees.

### **External Threats (Outsiders)**

External adversaries target AI systems in several ways:  
- **Data re-identification**: An example is the **Netflix Prize dataset**, where anonymized data was reverse-engineered to identify users.  
- **Adversarial attacks**: Input manipulations deceive AI models, such as a Tesla Model S being tricked into steering toward oncoming traffic.  
- **Processing externalization**: Large-scale AI models like ChatGPT require significant computational resources (**±3,640 PetaFLOP days**) and often depend on cloud services, introducing additional vulnerabilities.

---

## **2. Privacy-Enhancing Technologies (PETs)**

To mitigate these threats, **Privacy-Enhancing Technologies (PETs)** provide robust solutions:

### **Trusted Execution Environments (TEEs)**

- **Description**: Isolated execution spaces that encrypt data and code during runtime.  
- **Advantages**: Ensure that sensitive data remains invisible and protected during processing.  
- **Limitations**: TEEs currently lack GPU support, restricting their application in large-scale AI models.

### **Fully Homomorphic Encryption (FHE)**

- **Description**: Enables computations on encrypted data without decryption.  
- **Applications**: Secure inference on sensitive data like medical records.  
- **Challenges**: High computational demands, though efforts like DARPA DPRIVE aim to reduce latency to **<25 ms** for smaller models.

### **Federated Learning (FL)**

- **Description**: Facilitates collaborative model training by keeping raw data local.  
- **Advantages**:
    - Minimizes data leakage risks.  
    - Enables secure inter-organizational collaborations.  
- **Risks**: Vulnerable to inference attacks, such as gradient leakage.

---

## **3. Impacts and Critical Considerations**

### **Benefits of PETs**

1. **Comprehensive Protection**: Secures data in transit, at rest, and in use.  
2. **Regulatory Compliance**: Ensures adherence to privacy laws like GDPR and HIPAA.  
3. **Collaborative Innovation**: Supports joint model development among institutions handling sensitive data (e.g., hospitals).

### **Limitations of PETs**

1. **Performance**: Techniques like FHE significantly increase computational overhead.  
2. **Complexity**: Implementation requires advanced infrastructure and skilled personnel.  
3. **Compatibility**: TEEs lack support for modern hardware like GPUs, limiting scalability.

---

## **4. Key Insights and Practical Applications**

### **Combining PETs for Advanced Solutions**

- **Federated Learning + FHE**: Combine local data processing with encrypted computations for highly secure, collaborative AI training.  
- **TEEs + Differential Privacy**: Process sensitive data securely in isolation while adding noise to prevent individual inference.

### **Practical Scenarios**

#### **1. Secure Healthcare Collaboration**  
Hospitals can use FL to train diagnostic models across distributed datasets while using FHE to protect patient records during inference.

#### **2. Autonomous Driving**  
TEEs ensure that sensitive data from connected cars remains secure, while adversarial defenses protect AI systems from manipulation.

#### **3. Financial Fraud Detection**  
Banks can collaborate on fraud detection models without sharing customer data, leveraging a combination of FL and Secure Multiparty Computation (SMPC).

---

## **5. Final Thoughts**

As the adoption of AI systems continues to grow, securing these workflows with Privacy-Enhancing Technologies is no longer optional—it is essential. PETs not only safeguard sensitive data but also enable collaborations that were previously impossible due to privacy concerns. The future of secure AI lies in combining these technologies to address multifaceted threats, fostering a responsible and trust-driven AI ecosystem.
