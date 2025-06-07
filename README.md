
# SAFE-ORG: Simulation Architecture for Organizational Safety using FCMs

This repository accompanies the research article titled:

**"FCM-Based Agent Architecture for Organizational Safety Simulation– Part B: Model Implementation and Behavioral Validation"**

---

## 🔍 Project Summary

This repository contains the full implementation of a simulation-ready architecture for modeling organizational safety behavior using **Fuzzy Cognitive Maps (FCMs)** within a **Multi-Agent System (MAS)**. The model captures how internal organizational structures—termed **Behavior-Shaping Mechanisms (BSMs)**—influence safety-related behaviors at individual, group, and system levels.

The model is grounded in theoretical frameworks from organizational safety science and was validated through application to a real-world **aviation safety department**. Simulation results explore stability, sensitivity, and boundary behavior under uncertainty, offering a platform for safety analysis and system redesign.

---

## 🧠 Core Features

- **Multi-layered architecture:** Includes agent-level, group-level, organizational environment, and BSM layers
- **FCM-based reasoning:** Internal decision logic is driven by fuzzy causal networks
- **Simulation under uncertainty:** Incorporates random nodes and expert-informed fuzzy inputs
- **Behavioral validation:** Face validation against expert judgment and theoretical expectations
- **Sensitivity and robustness testing:** Includes structured analysis of BSM and random node influence

---

## 📁 Repository Structure

| File / Folder | Description |
|---------------|-------------|
| `developing_graph.py` | Builds the hierarchical FCM graph structure based on Excel inputs |
| `initial_test.py` | Simple test of model dynamics and simulation pipeline |
| `main_simulation.py` | Full simulation experiment for organizational behavior |
| `sensitivity_input_nodes.py` | Sensitivity analysis on input-level (BSM) nodes |
| `sensitivity_intermediate_nodes.py` | Sensitivity analysis on intermediate (group/SOE) nodes |
| `sensitivity_random_nodes.py` | Sensitivity analysis on random input variables |
| `boundary_behavior_analysis.py` | Verifies system behavior under extreme input scenarios |
| `face_validation.py` | Expert-based face validation of model outputs |
| `.xlsx` files | Input graphs and adjacency matrices for FCM construction |
| `.csv` files | Sample outputs from simulation runs (e.g., sensitivity results) |
| `LICENSE` | Academic use license – see below for terms |

---

## 🚀 How to Run

### 1. Install Required Libraries

You may install dependencies using:

```bash
pip install -r requirements.txt
```

Typical dependencies include:
- `numpy`
- `pandas`
- `networkx`
- `matplotlib`
- `openpyxl`

> *You can adapt the `requirements.txt` based on your specific environment.*

### 2. Execute Simulation Steps

```bash
python developing_graph.py             # Build the simulation graph
python initial_test.py                # Run a basic test
python main_simulation.py            # Full simulation execution
python face_validation.py            # Compare model to expert assessment
```

Run sensitivity or validation components as needed:

```bash
python sensitivity_input_nodes.py
python sensitivity_intermediate_nodes.py
python sensitivity_random_nodes.py
python boundary_behavior_analysis.py
```

### 3. View Results

Some scripts produce `.csv` outputs (e.g., sensitivity results), which can be analyzed using any data analysis tool.

---

## 🔒 License

This project is released under an **Academic and Non-Commercial Research License**:

> Use is restricted to academic and research purposes.  
> Commercial use or redistribution is prohibited without written permission.  
> A patent application is in process. For questions, contact:  
> **Ahmad Dehghan Nejad** – `dehghan.nejad@aut.ac.ir`

See the full terms in the [LICENSE](./LICENSE) file.

---

## 📄 Citation

This work is based on the following article (currently under review):

**Ahmad Dehghan Nejad (2025).**  
*FCM-Based Agent Architecture for Organizational Safety Simulation– Part B: Model Implementation and Behavioral Validation.*  
_Submitted to: Safety Science_

---

## 📌 Notes

- The repository is part of a broader research program on computational safety modeling.  
- For background architecture and theoretical foundations, refer to **Part A** .
- Appendix sections referenced in the paper are aligned with internal components in this repository.

---

## 📬 Contact

For collaboration, academic inquiry, or licensing questions, please contact:

**Ahmad Dehghan Nejad**  
dehghan.nejad@aut.ac.ir
