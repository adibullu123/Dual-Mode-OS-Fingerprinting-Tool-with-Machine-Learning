# Dual-Mode-OS-Fingerprinting-Tool-with-Machine-Learning

## 👥 Project Team
**Mentor:** Sruti Karumanchi — Project Lead, Cyber-Physical Systems Research, Purdue University  
**Group Members:** Aditya Yalamanchi, Abhijit Boro (IIIT Allahabad)

## 📦 Project Delivery
Developed and delivered to Rheinland-Pfälzische Technische Universität Kaiserslautern-Landau (RPTU), Germany
##

This research implements an advanced OS fingerprinting tool employing a hybrid approach, combining passive fingerprinting via ARP spoofing and active fingerprinting through manual TCP connections. Powered by a Random Forest classifier, the machine learning model extracts features from network packets for accurate OS identification. The tool's modular design offers flexibility in selecting passive or active methods. Validated through real-world scenarios, the tool demonstrates precision in identifying OS types and versions, providing actionable insights for network security enhancement. The integration of machine learning ensures adaptability in dynamic network environments, making it a valuable asset for security professionals navigating intricate network landscapes.


## ⚙️ Implementation Overview

This OS fingerprinting tool uses a hybrid approach combining **passive ARP spoofing** and **active TCP connections** to collect network data and classify operating systems using a machine learning model (Random Forest).

---

### 🔹 Mode 1: ARP Spoofing (Passive)

- Performs ARP spoofing to position the system as a man-in-the-middle on the network  
- Captures TCP packets in real time without direct interaction with target devices  
- Feeds captured packets into the machine learning model for OS classification  
- Maintains **anonymity**, as target devices are not aware of monitoring  

---

### 🔹 Mode 2: TCP Connection (Active)

- Establishes a TCP connection with target devices when passive capture is not possible  
- Generates TCP packets required for classification  
- Uses these packets as input to the machine learning model  
- **Trade-off:** Target device becomes aware of the connection (reduced anonymity)  

---

### 🧠 Key Idea

- **Primary method:** Passive ARP spoofing for real-time, stealthy data collection  
- **Fallback method:** Active TCP connection when required packets are unavailable  
- Ensures reliable OS fingerprinting even in low-traffic scenarios  

## Mode 1: Using ARP Spoofing

![Mode 1](https://github.com/adibullu123/Dual-Mode-OS-Fingerprinting-Tool-with-Machine-Learning/assets/97466499/c3aa58d6-470b-4ed9-a5dd-bf9f08a85e22)

## Mode 2: Using Manual TCP Connection

![Mode 2](https://github.com/adibullu123/Dual-Mode-OS-Fingerprinting-Tool-with-Machine-Learning/assets/97466499/f66b15ae-f90e-4c8d-a9e6-c7a75f08c399)

## 📊 Accuracy and Output Demonstration

The OS fingerprinting tool, which combines passive ARP spoofing and active TCP connection techniques with a Random Forest classifier, achieved an accuracy of **78.05%** in identifying operating systems of target devices within a network.

![Accuracy Output](https://github.com/user-attachments/assets/32c48e50-fd9e-4d6d-b68f-a72e52cc0681)




