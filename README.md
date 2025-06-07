# 🧠 SAFE-ORG: A Multi-Agent FCM-Based Framework for Organizational Safety Behavior Simulation

## 📘 Overview

**SAFE-ORG** is an advanced simulation framework designed to analyze and improve organizational safety behaviors through a **Multi-Agent Fuzzy Cognitive Mapping (FCM)** approach. Each agent (e.g., departments such as Aviation Safety, Industrial Safety, Safety Assurance) is modeled as a structured cognitive system that integrates:

- Causal relationships among behavioral variables
- Randomly assigned psychological and contextual conditions
- Organizational control mechanisms with dynamic impact and time delay modeling

The model supports simulation-based safety analysis at both operational and strategic levels and can be used for:
- Predictive scenario analysis
- Sensitivity analysis of safety mechanisms
- Identifying key leverage points for systemic improvement

This framework is particularly aligned with the principles of **Safety-II**, **Resilience Engineering**, and **High-Reliability Organizations (HROs)**.

---

## 🚀 How to Run the Jupyter Notebooks in Google Colab

Each `.ipynb` file corresponds to a different part of the simulation workflow:
- `Multi_Agent_Architecture_FCM_Developing_Graph.ipynb`: Graph generation from Excel matrices
- `main_simulation.ipynb`: Simulation execution and state vector evolution
- `sensitivity_input_nodes.ipynb`, `sensitivity_intermediate_nodes.ipynb`: Sensitivity analysis
- `face_validation.ipynb`, `boundary_behavior_analysis.ipynb`: Model validation and boundary case exploration

### ✅ Step-by-Step Instructions

1. **Open Colab**  
   Go to [https://colab.research.google.com](https://colab.research.google.com)

2. **Load the Notebook from GitHub**  
   - In Colab, click on `File → Open Notebook`
   - Select the **GitHub** tab
   - Enter the repository name:  
     ```
     Ahmaddenka/SAFE-ORG
     ```
   - Choose one of the `.ipynb` files to open

3. **Upload Supporting Files Manually**  
   Each notebook requires certain `.xlsx` files located in the `data/` directory of the GitHub repository.  
   You need to:
   - Download the required `.xlsx` files from GitHub
   - Upload them manually in Colab using the file upload interface or the `files.upload()` cell in the notebook

4. **Run the Code**  
   Click `Runtime → Run all` or run each cell manually after uploading the Excel files.

---

## 📁 File Structure

```plaintext
📂 SAFE-ORG/
│
├── *.ipynb              # Main simulation notebooks
├── data/
│   ├── agent_graph.xlsx
│   ├── Agent_Group_graph.xlsx
│   ├── SOE_graph.xlsx
│   └── ...              # Other Excel input matrices
├── output/
│   └── *.csv            # Exported simulation results (generated at runtime)
├── README.md
└── requirements.txt     # Dependencies for local execution (not needed in Colab)
```

---

## ⚙️ Notes

- No `.py` scripts are used in this version; all code has been migrated to Jupyter notebooks.
- The Excel files are not automatically downloaded in Colab—you must upload them before running.
- If you wish to run the notebooks locally, ensure all dependencies in `requirements.txt` are installed.
