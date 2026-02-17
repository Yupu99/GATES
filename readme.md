# Cost-Aware Workflow Scheduling with Guided-ES 

This repository contains the full research codebase on cost-aware dynamic workflow scheduling in cloud environments. It extends the original GATES system with **Guided-ES** (surrogate-guided Evolution Strategies) and includes:

- OpenAI-ES baseline
- Guided-ES with surrogate episode counts **S ∈ {1, 5, 10}**
- End-to-end training and evaluation pipelines
- Plotting, table generation, and statistical analysis scripts

Primary references:

- *GATES: Cost-aware Dynamic Workflow Scheduling via Graph Attention Networks and Evolution Strategy* (IJCAI-2025)
- *Cost-Aware Dynamic Cloud Workflow Scheduling Using Self-attention and Evolutionary Reinforcement Learning* (ICSOC-2024, Best Paper)

---

## Environment Setup

Conda (recommended):

```bash
conda env create -f environment.yml -n your-env-name
conda activate your-env-name
```

`environment.yml` matches the original paper environment as closely as possible. Python >= 3.8 is required.

---

## Quick Start

Train a baseline policy (OpenAI-ES):

```bash
python SSH_main.py
```

Evaluate a saved model:

```bash
python SSH_eval.py
```

---

## Guided-ES and OpenAI-ES Training Entry Points

Core training scripts:

- OpenAI-ES sweeps: `main_openai_es_sweep.py`
- Guided-ES sweeps: `main_guided_es_sweep.py`
- Guided-ES single run: `main_guided_es.py`

SLURM launchers (examples):

- Guided-ES: `run_guided*.sl`
- OpenAI-ES: `run_openai_es.sl`
- Comparisons: `run_compare_array.sl`, `run_compare_best_surr1.sl`

---

## Results, Plots, and Tables

### Plotting

```bash
python results_scripts/plot_compare.py \
  --glob "slurm_compare_*.out" \
  --min_ep 500 \
  --max_ep 500 \
  --outdir results_plots/compare
```

### Tables (mean rewards)

```bash
python results_scripts/make_compare_tables.py \
  --glob "slurm_compare_*.out" \
  --include_variants guided_surr1 \
  --min_ep 500 \
  --max_ep 500 \
  --outdir results_tables/compare
```

### Statistical Analysis (paired permutation test + CI + effect size)

```bash
python results_scripts/run_compare_stats.py \
  --glob "slurm_compare_*.out" \
  --guided_variants guided_surr1 \
  --min_ep 500 \
  --max_ep 500 \
  --outdir results_tables/compare
```

Outputs:

- `results_plots/compare/*.pdf`
- `results_tables/compare/*_table.tex`
- `results_tables/compare/stats_summary.{csv,tex}`

---

## Important Note on γ–Cost Trend and Reproducibility

In the original paper, we observed an empirical trend: **larger γ (looser deadlines) → lower average total cost**. This was obtained under a specific software environment and a specific set of trained policies.

In this codebase, the reward is:

```
reward = -(VM_cost + SLA_penalty)
```

and **γ is only used inside the environment** to set deadlines and compute SLA penalties. For any fixed policy, changing γ evaluates the same policy in a different MDP. The algorithm does **not** guarantee “larger γ ⇒ smaller total cost.”

Therefore, under different environments (PyTorch/Gym/CUDA versions) or random seeds, it is possible to observe the opposite trend. Please interpret γ–cost behaviour as **empirical and environment-dependent**, and use `environment.yml` for the best chance of reproducing the original results.

---

## Repository Structure (high-level)

- `main_guided_es.py`, `main_guided_es_sweep.py`: Guided-ES training
- `main_openai_es_sweep.py`: OpenAI-ES training
- `results_scripts/`: plotting, table, and statistical analysis utilities
- `run_*.sl`: SLURM job launchers
- `logs/`, `results/`, `results_plots/`, `results_tables/`: outputs

---

## License

This project is licensed under the Apache License.
