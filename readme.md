> ### ⚠️ **Important note on $\gamma$–cost trend and reproducibility**
>
> In the original paper, we reported an empirical trend that **larger $\gamma$ (looser deadlines) leads to lower average total cost**. These results were obtained under a specific conda environment and a particular set of trained policies.
>
> In this code, the reward is implemented as:
>
> `reward = -(VM_cost + SLA_penalty)`
>
> and **$\gamma$ is only used inside the environment to set deadlines and compute SLA penalties**. For any **fixed trained policy**, changing $\gamma$ means evaluating the *same* policy in **different** MDPs. The algorithm **does not mathematically guarantee** that “larger $\gamma$ ⇒ smaller total cost” for a fixed policy.
>
> As a result, under different software environments (e.g., different PyTorch / Gym / CUDA versions) or random seeds, it is possible, and in our recent runs observable, that **total cost increases when $\gamma$ increases**, even though:
>
> - the source code is unchanged, and  
> - the per-gamma performance is still reasonable and comparable to baselines.
>
> Please understand the $\gamma$–cost trend in the paper as **empirical behaviour of one set of runs under one specific environment**, not as a strict monotonicity guarantee of the algorithm. For the best chance of reproducing the original results, please replicate the original conda environment (see `environment.yml`) and use the same experimental protocol.

# Cost-aware Dynamic Workflow Scheduling via Deep Reinforcement Learning

This repository hosts the implementation of our research works on Cost-aware Dynamic Workflow Scheduling:

1. *GATES: Cost-aware Dynamic Workflow Scheduling via Graph Attention Networks and Evolution Strategy,*  
   **Thirty-Fourth International Joint Conference on Artificial Intelligence, IJCAI-2025.**

2. *Cost-Aware Dynamic Cloud Workflow Scheduling Using Self-attention and Evolutionary Reinforcement Learning,*  
   **Awarded Best Paper at International Conference on Service-Oriented Computing, ICSOC-2024.**

---

## 📦 Environment Setup

You can set up the environment using the following methods:

### Using Conda (Recommended)

```bash
conda env create -f environment.yml -n your-env-name
conda activate your-env-name
```

> `environment.yml` corresponds to the environment used in the original paper experiments as closely as possible.  
> Make sure your Python version is `>= 3.8`.

---

## 🚀 Quick Start

1. Run the main training program:

```bash
python SSH_main.py
```

2. Test a saved model:

```bash
python SSH_eval.py
```

---

## 📊 Example Output

After training and testing, results will be saved in the `logs/` directory, including: total cost, SLA violations / penalties, and VM fees, which can be used for visualization and evaluation.

---

## 🔁 Reproducibility, $\gamma$–cost behaviour, and original trend

Several users (and I myself in newer environments) have observed that, when running this code with a different conda environment (newer versions of PyTorch / Gym / CUDA, etc.), the behaviour of **total cost vs. $\gamma$** can differ from what is shown in the original paper. In particular, instead of the empirical trend:

> larger $\gamma$ (looser deadlines) → lower average total cost,

some runs show the opposite trend:

> larger $\gamma$ → larger average total cost,

even though the source code is the same.

The followings explain what the implementation actually does and how to interpret these results.

### 1. What the implementation actually does

In the logs produced by this repository, each test step contains:

* `current testing reward`
* `current VM cost`
* `current SLA penalty`
* `gamma`: $\gamma$

From the logs we always have:

```python
current testing reward = -(VM_cost + SLA_penalty)
```

and **$\gamma$ is only used inside the environment to set workflow deadlines and compute the SLA penalty**.

Consequences:

* For a **fixed trained policy** $\pi$, changing $\gamma$ changes the **environment / MDP** (through deadlines and penalties).
* The algorithm does **not mathematically guarantee** that “larger $\gamma$ ⇒ smaller (VM_cost + SLA_penalty)” for this fixed policy.

Therefore, the monotonic $\gamma$–cost trend in the paper should be understood as **empirical behaviour of the specific policies obtained under the original environment**, not as a hard constraint enforced by the code.

### 2. Why different environments can change the $\gamma$–cost trend

The training procedure (ES + deep RL) is:

* **Highly non-convex and stochastic**, and
* **Sensitive to small changes** in:
  * library versions (PyTorch / Gym / CUDA / cuDNN),
  * numeric kernels and parallelism,
  * random seeds and rollout timing.

Even with identical source code, a slightly different conda environment can lead to **different local optima**. And their **generalization behaviour across $\gamma$** (how they trade off VM cost vs. SLA penalty when deadlines change) can be **very different**:

* In the **original environment**, the trained policies happened to *exploit* looser deadlines effectively, yielding:

  > larger $\gamma$ → lower total cost (as reported in the paper).

* In any **newer environments** created by the submitted environment.yml, the trained policy may be very strong at $\gamma$ = 1 but does **not exploit larger $\gamma$ as effectively**, leading to:

  > larger $\gamma$ → larger total cost for that fixed policy.

Both behaviours are consistent with the implemented reward and environment; the code itself does **not** enforce $\gamma$–monotonicity.

### 3. Workflow size (S/M/L) and out-of-distribution generalization

The default training configuration focuses on:

* training at **$\gamma$ = 5**,
* on **S-size workflows**.

Evaluating the same policy on:

* **different $\gamma$** values (e.g., $\gamma$ in [1.0, 2.25]), and
* **larger** workflow sizes (M/L),

is a deliberate **out-of-distribution (OOD) generalization test**:

* Larger DAGs (M/L) contain **more tasks**, so **raw total cost** naturally increases even for simple baselines (a scale effect).
* Different $\gamma$ values change the deadline and penalty structure, which shifts the state distribution and leads it to a **entirely different MDP problem** seen by the policy.

In this OOD setting, it is expected that:

* **raw total cost increases from S → M → L** due to problem size, and
* the **$\gamma$–cost trend for each size may differ** from the patterns reported in the paper when the policy converges to a different local optimum under a different environment/MDP.

### 4. Original environment and archived experiment logs

To make the original results more transparent and reproducible, we provide:

* `environment.yml`: a conda environment file that matches the environment used in the original paper experiments as much as possible (Python / PyTorch / Gym / CUDA versions).

* `archived_experiment_logs`: original experiment logs for the $\gamma$–cost results reported in the paper, including:
  * the trained model checkpoints,
  * the corresponding testing logs.

These logs demonstrate that, **under the original environment and seeds**, the paper’s $\gamma$–cost trend were indeed produced by running this code.

If you want to approximate the original results as closely as possible, please:

1. Create the conda environment from `environment.yml`, and  
2. Follow the same training and evaluation protocol as in the paper (and in the example scripts).

If you still obtain different $\gamma$–cost behaviour as reported in the paper, this is expected to some extent due to the sensitivity of deep RL / ES training. In that case, we suggest focusing on **relative performance vs. baselines** and treating the $\gamma$–cost trend as **empirical, environment-dependent behaviour**, rather than as strict monotonicity guarantees.

---

## 📚 Citation

If you find this project useful for your research, please consider giving a star and citing the following papers in your future works:

```bibtex
@inproceedings{huang2022cost,
  title={Cost-aware dynamic multi-workflow scheduling in cloud data center using evolutionary reinforcement learning},
  author={Huang, Victoria and Wang, Chen and Ma, Hui and Chen, Gang and Christopher, Kameron},
  booktitle={International Conference on Service-Oriented Computing},
  pages={449--464},
  year={2022},
  organization={Springer}
}

@inproceedings{shen2024cost,
  title={Cost-Aware Dynamic Cloud Workflow Scheduling Using Self-attention and Evolutionary Reinforcement Learning},
  author={Shen, Ya and Chen, Gang and Ma, Hui and Zhang, Mengjie},
  booktitle={International Conference on Service-Oriented Computing},
  pages={3--18},
  year={2024},
  organization={Springer}
}

@inproceedings{ijcai2025p960,
  title={GATES: Cost-aware Dynamic Workflow Scheduling via Graph Attention Networks and Evolution Strategy},
  author={Shen, Ya and Chen, Gang and Ma, Hui and Zhang, Mengjie},
  booktitle={Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, {IJCAI-25}},
  publisher={International Joint Conferences on Artificial Intelligence Organization},
  pages={8635--8643},
  year={2025}
}
```

---

## 🙋‍♂️ Contact

If you have any questions or academic collaboration interests, feel free to reach out:

**GitHub:** [YaShen998](https://github.com/YaShen998) **Email:** [ya.shen@ecs.vuw.ac.nz](mailto:ya.shen@ecs.vuw.ac.nz)

---

## 🙏 Acknowledgements

We gratefully acknowledge the prior works by [Victoria Huang](https://niwa.co.nz/people/victoria-huang), [Chen Wang](https://niwa.co.nz/people/chen-wang), and [Yifan Yang](https://scholar.google.com/citations?user=dO8kmG4AAAAJ&hl=zh-CN), whose codes laid the foundation for this simulator. This work was also supported by the [AI-SCC & Big Data Group](https://ecs.wgtn.ac.nz/Groups/AISCC/WebHome) at Victoria University of Wellington.

---

## 📝 License

This project is licensed under the Apache License.
