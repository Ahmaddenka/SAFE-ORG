#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pyDOE')


# In[ ]:


import networkx as nx

# Load the graph from the file
all_network = nx.read_graphml('/content/all_network.graphml')
print("Graph has been loaded from all_network.graphml")


# # ✍️ Boundary Behavior Validation of the FCM Model
# To complete the behavioral validation of the Fuzzy Cognitive Map (FCM) model, we conducted a boundary behavior analysis to test whether the system produces logically consistent output responses under extreme input conditions. Specifically, we tested whether setting all external input nodes—both behavior-shaping parameters (BSPs) and random contextual factors—to their minimum (0.0) or maximum (1.0) activation values results in corresponding shifts in the final output toward the lower (−1.0) or upper (+1.0) bounds of the conceptual scale, respectively.
# 
# This type of validation is important because a properly designed FCM should exhibit bounded and monotonic behavior: higher input activations should not produce disproportionately lower output values, and vice versa (Papageorgiou & Salmeron, 2013). In nonlinear FCM systems, this behavior is not guaranteed analytically due to the potential influence of feedback loops and the nonlinearity of transition functions. Therefore, we employed a simulation-based approach to empirically verify whether the system respects this expected monotonicity.
# 
# The simulation was configured such that all root nodes (BSPs and randoms) were set to a fixed extreme value (either 0.0 or 1.0), while intermediate (non-root, non-output) nodes were randomly initialized. This decision was based on prior evidence showing that the FCM model is numerically robust to variations in intermediate node initialization. The model was simulated iteratively over a fixed number of time steps, and the activation levels of all nodes were recorded throughout the propagation process.
# 
# The results demonstrated that, in both test conditions, the final activation levels of the behavioral output nodes consistently converged toward the same side of the conceptual scale as the input (either near −1.0 or near +1.0). This confirms that the model respects boundary constraints and that the influence of inputs is directionally consistent, as expected in well-structured cognitive modeling systems (Kosko, 1986; Carvalho & Tomé, 2010).
# 
# This boundary test reinforces the model’s validity for use in organizational and decision-making simulations by confirming that it does not generate unstable or logically inconsistent behavior under extreme but theoretically plausible scenarios.
# 
# 📚 References (APA Format)
# Carvalho, J. P., & Tomé, J. A. B. (2010). Rule-based fuzzy cognitive maps—expressing time in qualitative terms. Computational Intelligence, 26(4), 261–287. https://doi.org/10.1111/j.1467-8640.2010.00366.x
# 
# Kosko, B. (1986). Fuzzy cognitive maps. International Journal of Man-Machine Studies, 24(1), 65–75. https://doi.org/10.1016/S0020-7373(86)80040-2
# 
# Papageorgiou, E. I., & Salmeron, J. L. (2013). A review of fuzzy cognitive maps research during the last decade. IEEE Transactions on Fuzzy Systems, 21(1), 66–79. https://doi.org/10.1109/TFUZZ.2012.2201727
# 
# 

# In[ ]:


# FCM Boundary Behavior Test Script – Self-Contained
# ---------------------------------------------------
# Objective:
# This script verifies that the FCM model respects boundary conditions:
# - When all external input nodes (BSPs + randoms) are set to 0.0 → outputs approach -1.0
# - When all external input nodes (BSPs + randoms) are set to 1.0 → outputs approach +1.0
# Intermediate nodes are initialized randomly, as prior analysis showed output is insensitive to them

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import re
import unicodedata

# --- Utility Functions ---
def clean_node_name(name):
    name = str(name).strip().lower()
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r'[\u200b-\u200f\u202a-\u202e\u00a0]', '', name)
    return name

def normalize_weight(w, original_min=0, original_max=5, target_min=0, target_max=1):
    if w is None:
        return 0
    w = float(w)
    return ((w - original_min) / (original_max - original_min)) * (target_max - target_min) + target_min

def graph_to_fcm(g, weight_property=None, normalize=True):
    num_vertices = len(g.nodes())
    fcm_matrix = np.zeros((num_vertices, num_vertices))
    node_index = {node: idx for idx, node in enumerate(g.nodes())}
    for source, target, data in g.edges(data=True):
        source_idx = node_index[source]
        target_idx = node_index[target]
        if weight_property and weight_property in data:
            weight = data[weight_property]
            if normalize:
                weight = normalize_weight(weight)
            fcm_matrix[source_idx, target_idx] = weight
        else:
            fcm_matrix[source_idx, target_idx] = 1.0
    return fcm_matrix

def triangular_membership(x, a, b, c):
    if x < a or x > c:
        return 0.0
    elif x == b:
        return 1.0
    elif x < b:
        return (x - a) / (b - a) if (b - a) != 0 else 0.0
    else:
        return (c - x) / (c - b) if (c - b) != 0 else 0.0

def fuzzify_bsp_input(value):
    x = float(value)
    return [
        max(0, 1 - 4 * x),
        max(0, 1 - abs(4 * x - 1)),
        max(0, 1 - abs(4 * x - 2)),
        max(0, 1 - abs(4 * x - 3)),
        max(0, 4 * x - 3) if x >= 0.75 else 0
    ]

def defuzzify(fuzzy_vector):
    weights = [-1.0, -0.5, 0.0, 0.5, 1.0]
    return sum(f * w for f, w in zip(fuzzy_vector, weights)) / sum(fuzzy_vector)

def nonlinear_transition(weighted_sum, total_weight, alpha=1.5):
    if total_weight == 0:
        return 0.0
    return np.tanh(alpha * (weighted_sum / total_weight))

# --- Initialization Function ---
def initialize_state(graph, bsp_val=0.0, random_val=0.0, intermediate_range=(-1, 1),
                     behavior_pattern=r"_behavior_\d+$", random_keyword="random"):
    init_state = {}
    bsp_nodes, random_nodes, intermediate_nodes, output_nodes = [], [], [], []

    for i, (node, data) in enumerate(graph.nodes(data=True)):
        name = clean_node_name(data.get('name', str(node)))
        label = clean_node_name(data.get('label', str(node)))
        is_random = random_keyword in name
        is_behavior = re.search(behavior_pattern, label)
        is_bsp = graph.in_degree(node) == 0 and not is_random

        if is_bsp:
            init_state[i] = fuzzify_bsp_input(bsp_val)
            bsp_nodes.append(i)
        elif is_random:
            init_state[i] = fuzzify_bsp_input(random_val)
            random_nodes.append(i)
        elif is_behavior:
            raw_val = 0.0
            init_state[i] = {
                'Low': triangular_membership(raw_val, -1.0, -1.0, 0.0),
                'Medium': triangular_membership(raw_val, -1.0, 0.0, 1.0),
                'High': triangular_membership(raw_val, 0.0, 1.0, 1.0)
            }
            output_nodes.append(i)
        else:
            init_state[i] = np.random.uniform(*intermediate_range)
            intermediate_nodes.append(i)

    return init_state, bsp_nodes + random_nodes, output_nodes

# --- Simulation ---
def fcm_simulation_fuzzy_trace(fcm_matrix, init_state, root_nodes, output_nodes, steps=10, alpha=1.5):
    num_nodes = fcm_matrix.shape[0]
    state_trace = []
    current_state = init_state.copy()
    for _ in range(steps):
        state_trace.append({i: current_state[i] for i in range(num_nodes)})
        next_state = current_state.copy()
        for i in range(num_nodes):
            if i in root_nodes:
                continue
            weighted_sum = 0.0
            total_weight = 0.0
            for j in range(num_nodes):
                w = fcm_matrix[j, i]
                if w == 0:
                    continue
                input_val = current_state[j]
                if isinstance(input_val, dict):
                    input_val = defuzzify(list(input_val.values()))
                elif isinstance(input_val, list):
                    input_val = defuzzify(input_val)
                weighted_sum += input_val * w
                total_weight += abs(w)
            next_state[i] = nonlinear_transition(weighted_sum, total_weight, alpha)
        current_state = next_state
    return state_trace

# --- Plot Function ---
def plot_node_traces(state_trace, output_nodes):
    num_steps = len(state_trace)
    num_nodes = len(state_trace[0])
    traces = {i: [] for i in range(num_nodes)}

    for t in range(num_steps):
        for i in range(num_nodes):
            val = state_trace[t][i]
            if isinstance(val, dict) or isinstance(val, list):
                traces[i].append(defuzzify(val.values() if isinstance(val, dict) else val))
            else:
                traces[i].append(val)

    plt.figure(figsize=(16, 8))
    for i in range(num_nodes):
        label = f"Node {i}"
        linestyle = "-" if i in output_nodes else "-"
        plt.plot(traces[i], linestyle, label=label)

    plt.title("FCM Node Evolution Under Extreme Input Conditions")
    plt.xlabel("Iteration")
    plt.ylabel("Activation Level")
    #plt.legend()
    plt.grid(True)
    plt.ylim(-1.1, 1.1)
    plt.tight_layout()
    plt.show()

# --- Run Test Function ---
def run_boundary_test(graph, extreme_value=1.0):
    fcm_matrix = graph_to_fcm(graph, weight_property="weight", normalize=True)
    init_state, root_nodes, output_nodes = initialize_state(
        graph, bsp_val=extreme_value, random_val=extreme_value
    )
    trace = fcm_simulation_fuzzy_trace(fcm_matrix, init_state, root_nodes, output_nodes, steps=15)
    plot_node_traces(trace, output_nodes)

# Example usage:
# run_boundary_test(your_graph, extreme_value=0.0)  # lower bound
# run_boundary_test(your_graph, extreme_value=1.0)  # upper bound


# In[ ]:


run_boundary_test(all_network, extreme_value=0.0)  # lower bound


# In[ ]:


run_boundary_test(all_network, extreme_value=1.0)  # upper bound


# # **Sensitivity analysis for root nodes**

# ✍️ Purpose and Logic of the Input Sensitivity Analysis
# This component of the simulation framework implements a causal-path-based sensitivity analysis tailored to the structure of the Fuzzy Cognitive Map (FCM) model. Unlike traditional variance-based methods such as Sobol, this approach does not rely on the assumption of mutual independence among inputs. This distinction is essential, as many organizational mechanisms (e.g., safety training, risk assessment, communication systems) are inherently interdependent—an assumption that violates the core requirements of Sobol-type methods (Saltelli et al., 2008; Kucherenko et al., 2012).
# 
# The primary goal is to determine the relative causal influence of each root input node—typically representing behavioral shaping mechanisms (BSPs)—on the final behavioral outputs. This is crucial in safety-critical organizational contexts, where interventions must be prioritized based on how much leverage a given mechanism has in shaping system behavior (Levinthal & Workiewicz, 2023).
# 
# The simulation follows a structured three-level loop. At the top level, multiple organizational scenarios are initialized using Latin Hypercube Sampling (LHS) to capture uncertainty across the full model, including random contextual variables and intermediate node states. Within each scenario, the model iteratively perturbs each root input node across a predefined range, keeping all other inputs fixed. For each perturbation, the FCM simulation is run, and the resulting behavioral output changes are measured. The output variation range (max - min) due to each input node is then recorded as its local sensitivity indicator in that scenario.
# 
# Averaging these sensitivity indicators across multiple scenarios yields a robust, path-aware estimate of each input’s potential to influence safety-related outcomes. Because the FCM architecture includes feedback loops, indirect interactions, and nonlinear transitions, this approach captures both direct and indirect effects—making it more suitable for complex organizational models than linear or independence-based alternatives such as Sobol (Stach et al., 2005; Papageorgiou & Froelich, 2017).
# 
# In summary, variance-based methods like Sobol are theoretically elegant but assume input independence and are structure-agnostic; they do not consider the causal topology of the system. In contrast, our approach leverages the internal architecture of the FCM to produce a more realistic and interpretable analysis that aligns with the systemic nature of organizational behavior modeling.
# 
# 📚 References (APA style)
# Kucherenko, S., Tarantola, S., & Annoni, P. (2012). Estimation of global sensitivity indices for models with dependent variables. Computer Physics Communications, 183(4), 937–946.
# 
# Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J., Gatelli, D., ... & Tarantola, S. (2008). Global Sensitivity Analysis: The Primer. Wiley.
# 
# Stach, W., Kurgan, L., Pedrycz, W., & Reformat, M. (2005). Genetic learning of fuzzy cognitive maps. Fuzzy Sets and Systems, 153(3), 371–401.
# 
# Papageorgiou, E. I., & Froelich, W. (2017). A review of applications of fuzzy cognitive maps in healthcare. Engineering Applications of Artificial Intelligence, 65, 289–302.
# 
# Levinthal, D. A., & Workiewicz, M. (2023). The architecture of organizational adaptation. Strategic Management Journal, 44(3), 601–621.
# 
# 

# In[ ]:


get_ipython().system('pip install pyDOE')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import pandas as pd

# ---------- Utility Functions ----------
def clean_node_name(name):
    name = str(name).strip().lower()
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r'[\u200b-\u200f\u202a-\u202e\u00a0]', '', name)
    return name

def normalize_weight(w, original_min=0, original_max=5, target_min=0, target_max=1):
    if w is None:
        return 0
    w = float(w)
    return ((w - original_min) / (original_max - original_min)) * (target_max - target_min) + target_min

def graph_to_fcm(g, weight_property=None, normalize=True):
    num_vertices = len(g.nodes())
    fcm_matrix = np.zeros((num_vertices, num_vertices))
    node_index = {node: idx for idx, node in enumerate(g.nodes())}
    for source, target, data in g.edges(data=True):
        source_idx = node_index[source]
        target_idx = node_index[target]
        if weight_property and weight_property in data:
            weight = data[weight_property]
            if normalize:
                weight = normalize_weight(weight)
            fcm_matrix[source_idx, target_idx] = weight
        else:
            fcm_matrix[source_idx, target_idx] = 1.0
    return fcm_matrix

def triangular_membership(x, a, b, c):
    if x < a or x > c:
        return 0.0
    elif x == b:
        return 1.0
    elif x < b:
        return (x - a) / (b - a) if (b - a) != 0 else 0.0
    else:
        return (c - x) / (c - b) if (c - b) != 0 else 0.0

def fuzzify_bsp_input(value):
    x = float(value)
    return [
        max(0, 1 - 4 * x),
        max(0, 1 - abs(4 * x - 1)),
        max(0, 1 - abs(4 * x - 2)),
        max(0, 1 - abs(4 * x - 3)),
        max(0, 4 * x - 3) if x >= 0.75 else 0
    ]

def defuzzify(fuzzy_vector):
    weights = [-1.0, -0.5, 0.0, 0.5, 1.0]
    return sum(f * w for f, w in zip(fuzzy_vector, weights)) / sum(fuzzy_vector)

def nonlinear_transition(weighted_sum, total_weight, alpha=1.5):
    if total_weight == 0:
        return 0.0
    return np.tanh(alpha * (weighted_sum / total_weight))

# ---------- Initialization ----------
def initialize_fixed_fcm_state(graph, bsp_inputs, intermediate_values,
                               behavior_pattern=r"_behavior_\d+$",
                               random_keyword="random"):
    init_state = {}
    random_nodes = []
    root_nodes = []
    output_nodes = []

    for i, (node, data) in enumerate(graph.nodes(data=True)):
        name = clean_node_name(data.get('name', str(node)))
        label = clean_node_name(data.get('label', str(node)))

        is_random = random_keyword in name
        is_bsp = name in bsp_inputs
        is_behavior = re.search(behavior_pattern, label)
        is_root = graph.in_degree(node) == 0

        if is_random:
            random_nodes.append(i)
            init_state[i] = 0  # will be overwritten in each simulation

        elif is_bsp:
            init_state[i] = bsp_inputs[name]

        elif is_behavior:
            output_nodes.append(i)
            raw_val = 0  # fixed
            init_state[i] = {
                'Low': triangular_membership(raw_val, -1.0, -1.0, 0.0),
                'Medium': triangular_membership(raw_val, -1.0, 0.0, 1.0),
                'High': triangular_membership(raw_val, 0.0, 1.0, 1.0)
            }

        else:
            init_state[i] = intermediate_values.get(i, 0.0)

        if is_root:
            root_nodes.append(i)

    return init_state, random_nodes, root_nodes, output_nodes

# ---------- Simulation ----------
def fcm_simulation_numeric(fcm_matrix, init_state, root_nodes, random_nodes, iterations=10, alpha=1.5):
    num_nodes = fcm_matrix.shape[0]
    current_state = init_state.copy()

    for _ in range(iterations):
        next_state = current_state.copy()
        for i in range(num_nodes):
            if i in root_nodes or i in random_nodes:
                continue
            weighted_sum, total_weight = 0.0, 0.0
            for j in range(num_nodes):
                w = fcm_matrix[j, i]
                if w == 0:
                    continue
                input_val = current_state.get(j, 0.0)
                if isinstance(input_val, dict):
                    input_val = defuzzify(list(input_val.values()))
                elif isinstance(input_val, list):
                    input_val = defuzzify(input_val)
                weighted_sum += input_val * w
                total_weight += abs(w)
            next_state[i] = nonlinear_transition(weighted_sum, total_weight, alpha)
        current_state = next_state
    return current_state


# --- Sensitivity Analysis of Root Nodes ---
def run_input_sensitivity_analysis(all_network):
    # Build FCM adjacency matrix
    fcm_matrix = graph_to_fcm(all_network, weight_property="weight", normalize=True)

    # Configuration
    num_scenarios = 20
    sim_per_input = 20
    input_range = (0, 1.0)
    random_range = (-1.0, 1.0)
    intermediate_range = (-1.0, 1.0)

    # Identify root nodes
    bsp_nodes = [clean_node_name(data.get("name", str(n))) for n, data in all_network.nodes(data=True)
                 if all_network.in_degree(n) == 0 and "random" not in clean_node_name(data.get("name", ""))]

    # Identify all root node indices
    root_node_indices = [i for i, (n, data) in enumerate(all_network.nodes(data=True))
                         if all_network.in_degree(n) == 0 and "random" not in clean_node_name(data.get("name", ""))]

    results = np.zeros((num_scenarios, len(root_node_indices)))

    for s in range(num_scenarios):
        # Step 1: Fixed BSP values (LHS sampled for all BSPs)
        bsp_lhs = lhs(len(bsp_nodes), samples=1)
        bsp_lhs = input_range[0] + (input_range[1] - input_range[0]) * bsp_lhs
        bsp_inputs = {bsp_nodes[i]: fuzzify_bsp_input(bsp_lhs[0][i]) for i in range(len(bsp_nodes))}

        # Step 2: Fixed random variable values for the scenario
        random_values = {}
        for i, (node, data) in enumerate(all_network.nodes(data=True)):
            name = clean_node_name(data.get("name", ""))
            if "random" in name:
                random_values[i] = np.random.uniform(*random_range)

        # Step 3: Fixed intermediate node values for the scenario
        intermediate_values = {
            i: np.random.uniform(*intermediate_range) for i in all_network.nodes()
        }

        # Step 4: Initial fixed state for the scenario
        init_state_base, random_nodes, root_nodes, output_nodes = initialize_fixed_fcm_state(
            all_network, bsp_inputs, intermediate_values
        )

        for i_idx, root_node in enumerate(root_node_indices):
            # For each root node, perturb it multiple times
            sim_outputs = []
            lhs_samples = lhs(1, samples=sim_per_input)
            lhs_samples = input_range[0] + (input_range[1] - input_range[0]) * lhs_samples

            for sim_i in range(sim_per_input):
                # Clone base init state
                init_state = init_state_base.copy()
                init_state[root_node] = fuzzify_bsp_input(lhs_samples[sim_i][0])  # perturb single input

                # Run simulation
                final_state = fcm_simulation_numeric(fcm_matrix, init_state, root_nodes, random_nodes)

                # Average over all output nodes
                out_vals = []
                for n in output_nodes:
                    val = final_state[n]
                    if isinstance(val, dict):
                        out_vals.append(defuzzify(list(val.values())))
                    elif isinstance(val, list):
                        out_vals.append(defuzzify(val))
                    else:
                        out_vals.append(val)
                sim_outputs.append(np.mean(out_vals))

            # Measure sensitivity of current input (e.g., output range)
            output_range_value = np.max(sim_outputs) - np.min(sim_outputs)
            results[s, i_idx] = output_range_value

    # --- Plot the sensitivity heatmap ---
    plt.figure(figsize=(12, 6))
    for s in range(num_scenarios):
        plt.plot(results[s], label=f"Scenario {s+1}")
    plt.xlabel("Input Root Nodes")
    plt.ylabel("Output Range (Sensitivity Indicator)")
    plt.title("Input Sensitivity Analysis Across Scenarios")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Summary Statistics (optional) ---
    df = pd.DataFrame(results, columns=bsp_nodes)
    print("\n📊 Average Sensitivity Across Scenarios:")
    print(df.mean().sort_values(ascending=False).round(4))

    return df
#df_input_sensitivity = run_input_sensitivity_analysis(all_network)


# **Visualization Function**

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

def plot_bar_sensitivity(df, save_path=None):
    """
    Plots a bar chart of mean sensitivity per input.
    """
    plt.figure(figsize=(20, 8), dpi=300)
    df.mean().sort_values(ascending=False).plot(kind='bar', color='skyblue')
    plt.title("Mean Sensitivity per Input", fontsize=18)
    plt.ylabel("Mean Output Range", fontsize=14)
    plt.xticks(rotation=90, fontsize=10)
    plt.grid(True, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_box_sensitivity(df, save_path=None):
    """
    Plots a boxplot showing the distribution of sensitivity values per input.
    """
    plt.figure(figsize=(20, 8), dpi=300)
    df.boxplot(rot=90)
    plt.title("Sensitivity Distribution per Input", fontsize=18)
    plt.ylabel("Output Range", fontsize=14)
    plt.grid(True, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()



# In[ ]:


import unicodedata
import re

df_input_sensitivity = run_input_sensitivity_analysis(all_network)

plot_bar_sensitivity(df_input_sensitivity)
plot_box_sensitivity(df_input_sensitivity)



# 
# # Model Validation through Expert-Based Fuzzy Behavioral Assessment
# To support the preliminary validation of the model’s predictive capacity, we conducted an expert-based fuzzy evaluation aimed at estimating the actual safety behavior profile of the safety department under study. This assessment was grounded in a conceptual framework informed by STAMP (System-Theoretic Accident Model and Processes; Leveson, 2004) and principles of adaptive control. Within this framework, safe behavior in an organizational control unit is operationally defined as the degree to which its members actively engage in sensing, actuating, and updating the internal mechanisms that maintain a valid and adaptive mental model of the system under control.
# 
# Given the absence of formalized performance indicators for departmental-level safety behavior, we employed a structured triangulation of expert judgments. The evaluation involved three independent raters: the department’s safety manager, a senior technical staff member, and the author of this study, who also serves as a full-time employee within the unit. The dual role of the author—as both researcher and embedded safety professional—provided a unique integrative perspective based on direct observation and longitudinal involvement.
# 
# Each evaluator rated the safety behavior of all nine personnel in the department using a five-point Likert scale (1 = Very Low to 5 = Very High). These ratings were then transformed into fuzzy membership vectors across three linguistic categories—Low, Medium, and High—through a triangular fuzzification process, in line with established linguistic decision analysis methods (Herrera & Herrera-Viedma, 2000). The resulting 27 fuzzy assessments (9 individuals × 3 raters) were averaged to form a single, composite fuzzy estimate of the department’s collective safety behavior.
# 
# This expert-derived fuzzy profile was then compared to the FCM model’s predicted behavioral output under the matching input scenario. The close qualitative alignment between the expert-based evaluation and the simulation result provides initial evidence of the model’s face validity. Furthermore, the use of fuzzy linguistic representation enabled a semantically consistent comparison between human judgment and model prediction, reinforcing the interpretability of the simulation outputs.
# 
# 

# # ✅ Final Self-Contained Script with Fuzzy Comparison, Visualization, and Robust Error Handling
# 
# 

# In[ ]:


get_ipython().system('pip install pyDOE')


# In[ ]:


# ======================================================
# 📘 Self-Contained FCM Simulation & Expert Comparison Tool (Robust and Visualized)
# ======================================================

# ✅ Import Required Libraries
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import unicodedata
import re
from pyDOE import lhs





# ----------------------------
# uclidean_distance
# ----------------------------

def euclidean_distance(vec1, vec2):
    """
    Compute Euclidean distance between two 3-level fuzzy vectors.
    Accepts both dict or list inputs.
    """
    if isinstance(vec1, dict):
        vec1 = [vec1.get('Low', 0), vec1.get('Medium', 0), vec1.get('High', 0)]
    if isinstance(vec2, dict):
        vec2 = [vec2.get('Low', 0), vec2.get('Medium', 0), vec2.get('High', 0)]
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

# ----------------------------
# 🧠 Expert Ratings Processing
# ----------------------------

def fuzzify_likert_1to5(val):
    """
    Convert a Likert scale value (1 to 5) into a fuzzy vector [Low, Medium, High].

    Returns:
        List of 3 floats representing degrees of membership to [Low, Medium, High].
    """
    if val == 1:
        return [1.0, 0.0, 0.0]  # completely Low
    elif val == 2:
        return [0.6, 0.4, 0.0]  # mostly Low
    elif val == 3:
        return [0.0, 1.0, 0.0]  # Medium
    elif val == 4:
        return [0.0, 0.4, 0.6]  # mostly High
    elif val == 5:
        return [0.0, 0.0, 1.0]  # completely High
    else:
        raise ValueError(f"Invalid Likert value: {val}. Expected 1 to 5.")


def process_expert_judgment(file_path):
    df = pd.read_excel(file_path)
    ratings = df.iloc[:, 1:]
    all_fuzzy_vectors = []
    for col in ratings.columns:
        for val in ratings[col]:
            fuzzy_vec = fuzzify_likert_1to5(val)
            all_fuzzy_vectors.append(fuzzy_vec)
    avg_fuzzy = np.mean(all_fuzzy_vectors, axis=0)
    defuzz_list = [defuzzify({'Low': v[0], 'Medium': v[1], 'High': v[2]}) for v in all_fuzzy_vectors]
    return avg_fuzzy, defuzz_list



# --------------------------
# 📦 Import Required Libraries
# --------------------------
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyDOE import lhs
import unicodedata
import re

# --------------------------
# 🔧 Preprocessing Functions
# --------------------------
def clean_node_name(name):
    """
    Normalize and clean node names for reliable matching.
    """
    name = str(name).strip().lower()
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r'[\u200b-\u200f\u202a-\u202e\u00a0]', '', name)
    return name

def fuzzify_bsp_input(value):
    """
    Convert a crisp or linguistic BSP input to a 5-level fuzzy vector.
    """
    if isinstance(value, str):
        value = value.strip().lower()
        mapping = {
            "very low": [1, 0, 0, 0, 0],
            "low":      [0, 1, 0, 0, 0],
            "medium":   [0, 0, 1, 0, 0],
            "high":     [0, 0, 0, 1, 0],
            "very high":[0, 0, 0, 0, 1]
        }
        return mapping.get(value, [0, 0, 1, 0, 0])
    else:
        x = float(value)
        return [
            max(0, 1 - 4 * x),
            max(0, 1 - abs(4 * x - 1)),
            max(0, 1 - abs(4 * x - 2)),
            max(0, 1 - abs(4 * x - 3)),
            max(0, 4 * x - 3) if x >= 0.75 else 0
        ]

def triangular_membership(x, a, b, c):
    """
    Calculate membership value of x in a triangular fuzzy set.
    """
    if x < a or x > c: return 0.0
    elif x == b: return 1.0
    elif x < b: return (x - a) / (b - a)
    else: return (c - x) / (c - b)

def fuzzify_output(x):
    """
    Convert a crisp value to a 3-level fuzzy output.
    """
    return {
        'Low': triangular_membership(x, -1, -1, 0),
        'Medium': triangular_membership(x, -1, 0, 1),
        'High': triangular_membership(x, 0, 1, 1)
    }

def defuzzify(fuzzy_vector):
    """
    Convert a fuzzy vector (Low/Medium/High or Very Low/.../Very High) to a crisp value.
    """
    mapping_3 = {'Low': -1, 'Medium': 0, 'High': 1}
    mapping_5 = {'Very Low': -1, 'Low': -0.5, 'Medium': 0, 'High': 0.5, 'Very High': 1}

    labels = set(fuzzy_vector.keys())
    if labels <= set(mapping_3.keys()):
        mapping = mapping_3
    elif labels <= set(mapping_5.keys()):
        mapping = mapping_5
    else:
        raise ValueError(f"Unrecognized fuzzy labels: {labels}")

    numerator = sum(mapping[k] * v for k, v in fuzzy_vector.items())
    denominator = sum(fuzzy_vector.values())
    return numerator / denominator if denominator != 0 else 0

# --------------------------
# 📊 FCM Structure and Initialization
# --------------------------
def graph_to_fcm(g, weight_property="weight", normalize=True):
    """
    Convert a NetworkX graph to an adjacency matrix.
    """
    nodes = list(g.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    W = np.zeros((len(nodes), len(nodes)))
    for u, v, d in g.edges(data=True):
        w = float(d.get(weight_property, 1))
        W[idx[u], idx[v]] = (w / 5.0) if normalize else w
    return W

def initialize_fcm_state_fuzzy_final(graph, bsp_inputs):
    """
    Initialize FCM node states, identifying random, root, and output nodes.
    """
    init_state = {}
    rand_nodes, root_nodes, out_nodes = [], [], []
    for i, (node, data) in enumerate(graph.nodes(data=True)):
        name = clean_node_name(data.get("name", str(node)))
        label = clean_node_name(data.get("label", str(node)))
        is_random = "random" in name
        is_behavior = re.search(r"_behavior_\d+$", label or "")
        is_bsp = name in bsp_inputs
        is_root = graph.in_degree(node) == 0
        if is_random:
            init_state[i] = np.random.uniform(-1, 1)
            rand_nodes.append(i)
        elif is_bsp:
            vec = bsp_inputs[name]
            init_state[i] = {k: float(v) for k, v in zip(["Very Low", "Low", "Medium", "High", "Very High"], vec)}
        elif is_behavior:
            out_nodes.append(i)
            init_state[i] = fuzzify_output(np.random.uniform(-1, 1))
        else:
            init_state[i] = np.random.uniform(-1, 1)
        if is_root: root_nodes.append(i)
    return init_state, rand_nodes, root_nodes, out_nodes

# --------------------------
# 🔁 Core FCM Simulation
# --------------------------
def nonlinear_transition(wsum, wtot, alpha=1.5):
    return np.tanh(alpha * wsum / wtot) if wtot != 0 else 0.0

def fcm_simulation(W, init_state, root_nodes, rand_nodes, out_nodes, n_iter=10):
    """
    Run one FCM simulation over time steps.
    """
    S = init_state.copy()
    for _ in range(n_iter):
        S_next = {}
        for i in range(W.shape[0]):
            if i in root_nodes or i in rand_nodes:
                S_next[i] = S[i]
                continue
            wsum, wtot = 0, 0
            for j in range(W.shape[0]):
                w = W[j, i]
                if w == 0: continue
                x = defuzzify(S[j]) if isinstance(S[j], dict) else S[j]
                wsum += w * x
                wtot += abs(w)
            res = nonlinear_transition(wsum, wtot)
            S_next[i] = fuzzify_output(res) if i in out_nodes else res
        S = S_next
    return S

# --------------------------
# 📈 LHS Sampling and Aggregation
# --------------------------
def determine_lhs_sample_size(n_rand):
    """Determine number of LHS samples based on random node count."""
    if n_rand <= 5: return 100
    elif n_rand <= 10: return 200
    elif n_rand <= 20: return 300
    else: return 500

def run_lhs_simulation(graph, W, bsp_inputs, rand_nodes, root_nodes, out_nodes, n_iter=10):
    d = len(rand_nodes)
    n_samples = determine_lhs_sample_size(d)
    lhs_samples = lhs(d, samples=n_samples) * 2 - 1
    fuzzy_agg_list = []
    defuzz_agg_list = []
    for s in lhs_samples:
        init_state, _, _, _ = initialize_fcm_state_fuzzy_final(graph, bsp_inputs)
        for idx, val in zip(rand_nodes, s):
            init_state[idx] = val
        final = fcm_simulation(W, init_state, root_nodes, rand_nodes, out_nodes, n_iter=n_iter)
        # Fuzzy aggregation
        fuzzy_vals = [final[i] for i in out_nodes]
        agg = {'Low': 0, 'Medium': 0, 'High': 0}
        for f in fuzzy_vals:
            for k in agg:
                agg[k] += f.get(k, 0)
        for k in agg:
            agg[k] /= len(fuzzy_vals)
        fuzzy_agg_list.append(agg)
        # Defuzzified aggregation
        defuzz_vals = [defuzzify(final[i]) for i in out_nodes]
        defuzz_agg_list.append(np.mean(defuzz_vals))
    return fuzzy_agg_list, defuzz_agg_list

# --------------------------
# 📊 Visualization & Summary
# --------------------------
def plot_fuzzy_bar(fuzzy_agg_mean):
    plt.figure(figsize=(6, 1.2))
    colors = ['red', 'orange', 'green']
    levels = ['Low', 'Medium', 'High']
    if isinstance(fuzzy_agg_mean, (list, np.ndarray)):
      fuzzy_agg_mean = dict(zip(levels, fuzzy_agg_mean))
    start = 0
    for lvl, color in zip(levels, colors):
        width = fuzzy_agg_mean[lvl]
        plt.barh(0, width, left=start, color=color, edgecolor='black')
        start += width
    plt.xlim(0, 1)
    plt.yticks([])
    plt.title("Fuzzy Aggregated Output")
    plt.show()

def plot_defuzzified_bar_normalized(defuzz_results):
    """
    Visualize defuzzified outputs mapped to [0,1] for visual consistency with fuzzy outputs.
    """
    # Map defuzzified results from [-1,1] → [0,1]
    norm_results = [(x + 1) / 2 for x in defuzz_results]
    mean = np.mean(norm_results)
    low = np.percentile(norm_results, 5)
    high = np.percentile(norm_results, 95)

    plt.figure(figsize=(6, 1.2))
    plt.barh(0, 1, left=0, color='lightgray', edgecolor='black')  # background bar
    plt.plot(mean, 0, 'ko', label='Mean')
    plt.errorbar(mean, 0, xerr=[[mean - low], [high - mean]], fmt='o', color='black', capsize=5)

    plt.xlim(0, 1)
    plt.yticks([])
    plt.xlabel("Defuzzified score (rescaled to 0–1)")
    plt.title("Normalized Defuzzified Output with 90% CI")
    plt.tight_layout()
    plt.show()

def summarize_defuzzified(defuzz_results):
    """
    Generate a summary table with uncertainty metrics and definitions.
    """
    summary = {
        "Mean": np.mean(defuzz_results),
        "Std Dev": np.std(defuzz_results),
        "Min": np.min(defuzz_results),
        "Max": np.max(defuzz_results),
        "5th Percentile": np.percentile(defuzz_results, 5),
        "95th Percentile": np.percentile(defuzz_results, 95),
        "95% CI Width": np.percentile(defuzz_results, 95) - np.percentile(defuzz_results, 5)
    }
    definitions = {
        "Mean": "Average system-level behavior score.",
        "Std Dev": "Standard deviation indicating variation across samples.",
        "Min": "Lowest observed output in the simulations.",
        "Max": "Highest observed output in the simulations.",
        "5th Percentile": "Lower bound of 90% confidence interval.",
        "95th Percentile": "Upper bound of 90% confidence interval.",
        "95% CI Width": "Spread of the 90% confidence range (sensitivity width)."
    }
    df = pd.DataFrame.from_dict(summary, orient='index', columns=["Value"])
    df["Definition"] = df.index.map(definitions)
    return df


# --------------------------
# 🧠 Final Simulation Function
# --------------------------
def run_full_simulation(graph_file, excel_file):
    G = nx.read_graphml(graph_file)
    df = pd.read_excel(excel_file)
    df.columns = df.columns.str.strip().str.lower()
    bsp_inputs = {clean_node_name(row['name']): fuzzify_bsp_input(row['value']) for _, row in df.iterrows()}
    W = graph_to_fcm(G)
    init_state, rand_nodes, root_nodes, out_nodes = initialize_fcm_state_fuzzy_final(G, bsp_inputs)
    fuzzy_results, defuzz_results = run_lhs_simulation(G, W, bsp_inputs, rand_nodes, root_nodes, out_nodes)
    fuzzy_avg = {
        'Low': np.mean([f['Low'] for f in fuzzy_results]),
        'Medium': np.mean([f['Medium'] for f in fuzzy_results]),
        'High': np.mean([f['High'] for f in fuzzy_results])
    }
    plot_fuzzy_bar(fuzzy_avg)
    plot_defuzzified_bar_normalized(defuzz_results)
    summary_df = summarize_defuzzified(defuzz_results)
    return fuzzy_avg, defuzz_results, summary_df



# ----------------------------
# ✅ Final Comparison & Report
# ----------------------------

def compare_model_to_expert(model_vec, expert_vec, expert_defuzz_list):
    d_model = defuzzify(model_vec)
    d_expert = defuzzify({'Low': expert_vec[0], 'Medium': expert_vec[1], 'High': expert_vec[2]})
    euclid = euclidean_distance(model_vec, {'Low': expert_vec[0], 'Medium': expert_vec[1], 'High': expert_vec[2]})
    print("\n🔍 Comparison Summary:")
    print(f"Model defuzzified: {d_model:.3f} | Expert defuzzified: {d_expert:.3f}")
    print(f"Absolute difference: {abs(d_model - d_expert):.3f}")
    print(f"Euclidean distance (fuzzy vectors): {euclid:.3f}")



# In[ ]:


# Befor Executeing Upload "all_network.graphml", "fcm_input_template.xlsx", and "expert_ratings.xlsx"

print("\n Model Output:")
fuzzy_output, defuzz_values, summary_table = run_full_simulation("all_network.graphml", "fcm_input_template.xlsx")
print("\n Expert Result:")
avg_fuzzy, defuzz_list = process_expert_judgment("expert_ratings.xlsx")
plot_defuzzified_bar_normalized(defuzz_list)
plot_fuzzy_bar(avg_fuzzy)
compare_model_to_expert(fuzzy_output, avg_fuzzy, defuzz_list)


# In[ ]:


import pandas as pd
import numpy as np

def compute_sdi(file_path):
    """
    Compute SDI (Standard Deviation Index) for expert ratings.

    Parameters:
    - file_path: Path to Excel file with expert Likert ratings (1–5).
                 First column = individual/team member name.
                 Remaining columns = ratings from each expert.

    Returns:
    - sdi (float): Average standard deviation across raters.
    - stds (Series): Standard deviation per individual.
    """

    df = pd.read_excel(file_path)
    ratings = df.iloc[:, 1:]  # Exclude name column
    stds = ratings.std(axis=1)
    sdi = stds.mean()

    print("\n🔍 Inter-Rater Disagreement Summary")
    print(f"SDI (Average Std Dev): {sdi:.3f}")
    print(f"Minimum SD: {stds.min():.3f}")
    print(f"Maximum SD: {stds.max():.3f}")

    # Interpretation
    if sdi < 0.5:
        level = "🟢 High agreement among experts"
    elif sdi < 1.0:
        level = "🟡 Moderate agreement among experts"
    else:
        level = "🔴 Low agreement or high disagreement among experts"

    print(f"Interpretation: {level}")

    return sdi, stds

# Example:
# sdi_val, stds_per_person = compute_sdi("expert_judgments.xlsx")


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# # Expert-Based Validation via Fuzzy Vector Comparison
# To provide a quantitative basis for validating the predictive accuracy of the FCM simulation model, we implemented a comparison framework that operates on the fuzzy output representations of both the model and expert judgment. The comparison method supports two levels of analysis: defuzzified scalar distance and fuzzy vector similarity.
# 
# Each fuzzy vector—representing the final collective safety behavior—is expressed using a three-level linguistic scale: Low, Medium, and High. These vectors are normalized membership distributions derived either from the FCM output or from aggregated expert evaluations. To enable numeric comparison, both vectors are defuzzified using a standard linear transformation, assigning weights of -1.0, 0.0, and +1.0 to the respective membership levels (Herrera & Herrera-Viedma, 2000). The resulting scalar values provide a crisp estimate of the overall behavioral quality.
# 
# To assess proximity between the two fuzzy profiles beyond scalar aggregation, we also compute the Euclidean distance between the model-generated and expert-derived fuzzy vectors. This metric preserves the internal distributional structure of the fuzzy ratings, offering a finer-grained perspective on the similarity between the two assessments.
# 
# The outputs of the comparison module include:
# 
# Defuzzified values for both model and expert vectors
# 
# Absolute difference between defuzzified values
# 
# Euclidean distance between the full fuzzy vectors
# 
# This dual-mode comparison facilitates both interpretability and sensitivity analysis, allowing for validation of the FCM model's predictive realism in a semantically consistent fuzzy framework.
# 
# 📚 References (APA Style)
# Herrera, F., & Herrera-Viedma, E. (2000). Linguistic decision analysis: Steps for solving decision problems under linguistic information. Fuzzy Sets and Systems, 115(1), 67–82. https://doi.org/10.1016/S0165-0114(99)00023-3
# 
# 
