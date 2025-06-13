# AI-Driven Threat Simulation (ACT-Bench)

## 1. **Topic Name:**
**AI-Driven Threat Simulation for Intrusion Detection Systems (IDS) and Intrusion Prevention Systems (IPS)**

---

# AI-Enhanced IDS/IPS Benchmarking Framework

## Overview of the Project's Purpose
The **AI-Enhanced IDS/IPS Benchmarking Framework** is designed to assess and enhance the resilience of Intrusion Detection Systems (IDS) and Intrusion Prevention Systems (IPS) against dynamic, evolving cyber threats. Traditional benchmarking methods rely on static traffic patterns and predefined rules, which fail to represent the sophisticated tactics employed by modern attackers. This project introduces AI-driven simulations and advanced evaluation metrics to create a realistic and adaptive testing environment for IDS/IPS systems.

---
## 2. Key Features

### 1. AI-Driven Threat Simulation
- **Purpose**: To simulate realistic and adaptive cyber threats for testing IDS/IPS systems.  
- **Components**:  
  - **Generative Adversarial Networks (GANs)**:  
    - **Generator**: Creates synthetic attack patterns designed to mimic real-world cyber threats.  
    - **Discriminator**: Identifies real versus synthetic attack data, pushing the generator to create increasingly realistic and complex threats.  
  - **Dynamic Attack Patterns**: Produces polymorphic (changing) attack types to challenge signature-based IDS/IPS detection mechanisms.  

### 2. Reinforcement Learning (RL)
- **Purpose**: To introduce a learning-based adaptive mechanism where simulated attackers evolve to bypass IDS/IPS defenses.  
- **Mechanism**:  
  - RL agents act as attackers, receiving feedback based on the IDS/IPS's detection performance.  
  - Agents continuously refine their attack strategies to adapt to the system's defenses, mimicking real-world adversarial behaviors.  
  - Real-time adaptability is modeled to simulate attackers learning and evading detection.

---

## Simulation of Realistic Attacks
- **Adaptive and Evolving Patterns**: The system simulates attack types such as:
  - Distributed Denial of Service (DDoS)
  - Data Exfiltration
  - Ransomware Propagation
  - Zero-Day Exploits  
- **Polymorphic Attacks**: Dynamic behaviors force IDS/IPS systems to detect threats that change over time, challenging traditional rule-based detection.  

---

This framework introduces a comprehensive benchmarking solution, leveraging AI technologies like GANs and RL to simulate real-world cyber threats. This ensures IDS/IPS systems are prepared for the complexities of modern cybersecurity challenges.

---

## 3. **Installation Instructions:**

### **Prerequisites:**
- Python 3.10 (Recommended)
- PyTorch 
- Pandas 
- Other Python dependencies mentioned in requirement.txt

### **Step-by-Step Installation:**

1. **Clone the Repository:**

   Clone the repository to your local machine using Git:
   ```bash
   git clone "Paste URL"
   ```

2. **Create a Virtual Environment:**

   Create a virtual environment to manage project dependencies:

   ```bash
   python -m venv venv
   ```

   - For Linux/Mac

   ```bash
   source venv/bin/activate
   ```

   - For Windows

   ```bash
   venv\Scripts\activate
   ```
3. **Install Dependencies:**

    Install the required dependencies using `pip` by running the following command:

    ```bash
    pip install -r requirements.txt
    ```
