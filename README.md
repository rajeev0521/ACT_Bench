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

### 3. Latency vs. Action Responsiveness (LAR) Score
- **Purpose**: Measures the trade-off between fast threat detection and minimal impact on legitimate traffic.  
- **Components**:  
  - **Dynamic Balancing**: Uses a Deep Q-Network (DQN) to optimize IDS/IPS configurations for high detection accuracy with minimal latency.  
  - **Predictive Latency Modeling**: Leverages models like Decision Trees or LSTMs to forecast network latency spikes and enable proactive adjustments to IDS/IPS settings.  
- **Impact**: Ensures IDS/IPS systems respond swiftly to threats without degrading network performance.

---

## Simulation of Realistic Attacks
- **Adaptive and Evolving Patterns**: The system simulates attack types such as:
  - Distributed Denial of Service (DDoS)
  - Data Exfiltration
  - Ransomware Propagation
  - Zero-Day Exploits  
- **Polymorphic Attacks**: Dynamic behaviors force IDS/IPS systems to detect threats that change over time, challenging traditional rule-based detection.  

---

## Evaluation Metrics
- **Throughput**: Measures the data-processing capacity of IDS/IPS systems under various loads.  
- **Detection Accuracy**: Assesses the system's ability to correctly identify malicious traffic.  
- **Latency**: Evaluates the delay introduced by the IDS/IPS while inspecting traffic.  
- **Packet Drop Rate**: Tracks the percentage of dropped packets under high traffic conditions.  
- **LAR Score**: A composite metric to balance detection accuracy and network latency.

---

This framework introduces a comprehensive benchmarking solution, leveraging AI technologies like GANs and RL to simulate real-world cyber threats while maintaining a fine balance between accuracy and latency through the LAR Score. This ensures IDS/IPS systems are prepared for the complexities of modern cybersecurity challenges.

---

## 3. **Installation Instructions:**

### **Prerequisites:**
- Python 3.7 or higher
- PyTorch (for AI model training)
- Pandas (for data processing)
- Other Python dependencies

### **Step-by-Step Installation:**

1. **Clone the Repository:**

   Clone the repository to your local machine using Git:
   ```bash
   git clone https://github.com/your-repo-name/ai-threat-simulation.git
   cd ai-threat-simulation

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



# Project Status Update: AI-Enhanced IDS/IPS Benchmarking Framework

## Current Progress

### 1. AI-Driven Threat Simulation: **80% Completed**
- **Overview**: This module is the backbone of the project, simulating realistic and adaptive attack patterns to evaluate IDS/IPS resilience.
- **Key Achievements**:
  - **Generative Adversarial Networks (GANs)**:
    - Initial generator and discriminator models are implemented and generating attack patterns.
    - Basic polymorphic attack simulation is operational, challenging signature-based detection systems.
  - **Reinforcement Learning (RL)**:
    - RL environment has been designed to adapt attack strategies dynamically based on IDS/IPS responses.
- **Pending Tasks**:
  - Fine-tune GANs on real dataset to produce more complex and realistic attack patterns.
  - Expand RL agent capabilities for advanced attack tactics like zero-day exploits.
  - Integrate GAN and RL modules for a seamless simulation pipeline.

---

### 2. Latency vs. Action Responsiveness (LAR) Score: **Not Started**
- **Overview**: This module will measure the balance between IDS/IPS detection speed and network latency. The LAR Score ensures that security measures are robust without compromising network performance.
- **Planned Work**:
  - **Dynamic Balancing**: Implement a Deep Q-Network (DQN) to optimize detection accuracy with minimal latency.
  - **Predictive Latency Modeling**: Train a regression model to forecast latency spikes and fine-tune IDS/IPS configurations proactively.

---

## Next Steps
1. **Finalize AI-Driven Threat Simulation**:
   - Refine GAN and RL models for enhanced attack realism and adaptability.
   - Complete module integration and end-to-end testing.
2. **Begin LAR Score Implementation** :
   - Initiate the design and development of DQN-based balancing and latency prediction models.
   - Establish baseline metrics to validate LAR Score performance.

---

## Summary
The **AI-Driven Threat Simulation** module is 80% complete and nearing finalization. Once completed, focus will shift to implementing the **LAR Score**, which will add a critical layer of performance evaluation by balancing detection accuracy and network efficiency. The project remains on track for delivery by.
