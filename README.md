# SAFE-ORG: Fuzzy Simulation Framework for Organizational Safety Behavior

This repository contains the Jupyter Notebook implementation of a simulation framework designed to analyze and optimize organizational safety behavior using a fuzzy cognitive mapping approach.

All computational modules, validation steps, and sensitivity analyses are presented in executable `.ipynb` files.

---

## 🚀 How to Run

### 1. Run on Google Colab (Recommended for Reviewers)

If you want to explore the notebooks **without installing anything**, follow these steps:

1. Open [Google Colab](https://colab.research.google.com).
2. Go to the **GitHub** tab.
3. Paste the repository URL:
   ```
   https://github.com/Ahmaddenka/SAFE-ORG
   ```
4. Choose the notebook you want to explore (e.g., `developing_graph.ipynb` or `main_simulation.ipynb`).
5. At the beginning of each notebook, **run the cell to install required packages**:

```python
!pip install -r https://raw.githubusercontent.com/Ahmaddenka/SAFE-ORG/main/requirements.txt
```

> **Note:** Installing dependencies may take 1–2 minutes.

---

### 2. Run Locally (Advanced Users)

#### Prerequisites
- Python ≥ 3.8  
- Jupyter Notebook / JupyterLab  
- All dependencies listed in `requirements.txt`

#### Installation

```bash
git clone https://github.com/Ahmaddenka/SAFE-ORG.git
cd SAFE-ORG
pip install -r requirements.txt
jupyter notebook
```

---

## 📁 File Structure

- **`*.ipynb` files**: Main simulation components (graph generation, sensitivity analysis, boundary behavior, etc.)
- **`.xlsx` files**: Input adjacency matrices and configurations
- **`.graphml` files**: Pre-generated graph representations (optional use)
- **`.csv` files**: Simulation outputs saved during runtime

> All file paths are **relative** and designed to work from the repository's root.

---

## 📌 Key Notebooks

| Notebook                          | Purpose                                     |
|----------------------------------|---------------------------------------------|
| `developing_graph.ipynb`         | Build the simulation network graph          |
| `main_simulation.ipynb`          | Perform full-scale safety behavior simulation |
| `sensitivity_input_nodes.ipynb`  | Sensitivity analysis on input (root) nodes  |
| `sensitivity_intermediate_nodes.ipynb` | Sensitivity on intermediate nodes     |
| `sensitivity_random_nodes.ipynb` | Sensitivity on random-effect nodes          |
| `boundary_behavior_analysis.ipynb` | Boundary condition exploration            |
| `face_validation.ipynb`          | Expert validation of model behavior         |
| `initial_test.ipynb`             | Initial run to verify model initialization  |

---

## 📂 Input Files

All required Excel input files are stored in the main directory:
- `SOE_graph.xlsx`, `group_graph.xlsx`, etc. contain the structure of each subgraph.
- `all_network.graphml` is an optional full network file (used in some notebooks).

> These files are automatically loaded by each notebook. **No manual file movement is needed.**

---

## 🔄 Output Files

Some notebooks save `.csv` results (e.g., behavior state vectors, sensitivity metrics) into the root directory.  
These outputs can be directly downloaded from Colab or analyzed locally.

---

## 🧪 Reproducibility Notes

- The simulation involves fuzzy and stochastic components.
- For exact reproducibility, fixed random seeds can be added to each notebook if needed.

---

## 📄 License

This code and repository are released under the **Academic Research License**:

- Redistribution and use for **non-commercial, academic purposes** are permitted.
- Citation of the original paper (when published) is **required**.

Author: **Ahmad Dehghan Nejad**  
Contact: [dehghan.nejad@aut.ac.ir](mailto:dehghan.nejad@aut.ac.ir)

---

## 📘 Citation

The article describing this simulation framework is currently under review for publication in *Safety Science*.  
Please cite the GitHub repository until the DOI of the paper is available:

```text
Ahmad Dehghan Nejad. SAFE-ORG: Simulation Framework for Fuzzy Analysis of Organizational Safety Behavior. GitHub Repository: https://github.com/Ahmaddenka/SAFE-ORG
```
