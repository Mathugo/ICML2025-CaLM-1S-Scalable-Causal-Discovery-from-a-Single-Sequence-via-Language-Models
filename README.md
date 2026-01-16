# ICML2025: CaLM-1S - Scalable Causal Discovery from a Single Sequence via Language Models

This repository contains the official implementation of **CaLM-1S**.

**CaLM-1S** is a method for **One-Shot Causal Discovery** that repurposes pre-trained Language Models (LMs) as density estimators. Unlike traditional iterative constraint-based methods (e.g., PCMCI) that scale quadratically with vocabulary size, CaLM-1S leverages **Vectorized Conditional Mutual Information (CMI)** to infer causal graphs from a single observed sequence in a parallelized forward pass.

## 🚀 Key Features

* **One-Shot Discovery:** Recovers causal structure from a single sequence $s$.
* **Massive Scalability:** Handles vocabularies $|\mathcal{X}| > 20,000$ (e.g., Vehicle Diagnostics, IT Logs).
* **Vectorized Inference:** Replaces sequential CI tests with parallelized tensor operations on GPUs.
* **End-to-End Pipeline:** From raw sequence logs $\to$ Instance Time Graph $\to$ One-Shot Summary Graph.

## 🛠️ Installation

We recommend using a virtual environment. The core dependencies are `torch`, `transformers`, and `accelerate`.

```bash
conda create -n calm-1s python=3.10
conda activate calm-1s

pip install -r requirements.txt

```

**requirements.txt:**

```text
torch>=2.0.0
transformers==4.26.1
accelerate>=0.15.0
numpy
seaborn
matplotlib
scikit-learn
datasets

```

## 📂 Project Structure

```text
├── data/                   # Example sequences / Diagnostic logs
├── models/                 # Pre-trained SCM models (or HF path)
├── src/
│   ├── calm_1s.py          # Core OneShotCausalDiscovery class
│   ├── evaluator.py        # TypeLevelEvaluator and plotting tools
│   ├── sampling.py         # Ancestral and multinomial sampling logic
│   └── utils.py            # Helper functions (extract_causes, etc.)
├── notebooks/
│   ├── experiment.ipynb          # Main experiment notebook (Reproduction)
│   └── training.ipynb            # Training notebook (Reproduction)
└── README.md

```

## ⚡ Quick Start

Here is a minimal example of running CaLM-1S on a single sequence using a pre-trained Transformer.

```python
import torch
from transformers import LlamaForCausalLM
from src.calm_1s import OneShotCausalDiscovery, extract_causes
from src.evaluator import TypeLevelEvaluator

# 1. Load Pre-trained Language Model (The "Physics" Engine)
model_path = 'scm_pretraining_tfx/event_type_with_m6_L64...'
model = LlamaForCausalLM.from_pretrained(model_path).to('cuda')

# 2. Configuration for Vectorized Inference
params = {
    'sampling': {
        'type': 'naive', 
        'ancestral': True,
        'value': 128,   # Number of Monte Carlo particles (N)
        'context': 4,   # Context window
    },
    'threshold': {'type': 'static', 'value': 0.005},
    'full': True        # Compute full CMI matrix
}

# 3. Initialize & Run CaLM-1S
# 'logits_key' ensures compatibility with different HF models
oneshot_cd = OneShotCausalDiscovery(model, params, dataset, logits_key='logits')
oneshot_cd.prepare()

# Returns the raw CMI matrix as the Instance Causal Graph (batch, seq_len, seq_len)
batch_out, cmi_matrix = oneshot_cd.run()

# 4. Pruning & Projection (Instance -> Summary Graph)
# Extract causal links and project to Event Types
causal_links = extract_causes(cmi_matrix, batch_out, params)

print(f"Discovered {len(causal_links)} causal edges in the summary graph.")

```

## 🧠 Methodology

### Overview

CaLM-1S operates in two distinct phases, separating density estimation from structure learning:

1. **Phase 1: Pre-training (Density Estimation)**
* We train a standard Causal Transformer (e.g., Llama, GPT) on domain-specific event logs.
* **Objective:** Learn $P(X_t | X_{<t})$.

2. **Phase 2: One-Shot Discovery (Inference)**
* **Input:** A single sequence $s$.
* **Vectorized CMI:** Instead of $O(L^2)$ forward passes, we construct a tensor of intervened contexts and compute Conditional Mutual Information in parallel.
* **Pruning:** We threshold the CMI matrix to obtain the **Instance Time Causal Graph** $\mathcal{G}_{t,s}$.
* **Projection:** We project temporal edges onto event types to recover the **One-Shot Summary Graph** $\mathcal{G}_s$.
  

> **Note:** As shown in the paper, while $\mathcal{G}_{t,s}$ is a DAG, the projected summary graph $\mathcal{G}_s$ correctly recovers cycles (feedback loops) if the sequence  contains temporal recurrences of the events.
