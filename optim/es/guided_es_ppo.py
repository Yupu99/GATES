
"""
    Guided-ES with PPO Surrogate Gradient
    PPO is used only to compute a gradient direction for updating the ES subspace U.
    Actor parameters are updated exclusively by ES; PPO only updates the Critic (value_head).
"""

from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from optim.base_optim import BaseOptim
from utils.policy_dict import agent_policy
from utils.torch_util_01 import get_flatten_params, set_flatten_params


class GuidedESWithPPO(BaseOptim):
    def __init__(self, config):
        super(GuidedESWithPPO, self).__init__()
        self.name = config["name"]
        self.sigma_init = config["sigma_init"]
        self.sigma_curr = self.sigma_init
        self.sigma_decay = config["sigma_decay"]
        self.learning_rate = config["learning_rate"]
        self.reinforce_learning_rate = config["reinforce_learning_rate"]
        self.population_size = config["population_size"]
        self.reward_shaping = config['reward_shaping']
        self.reward_norm = config['reward_norm']

        # Guided-ES hyper-parameters
        self.subspace_dim = config.get("subspace_dim", 20)
        self.alpha_min = config.get("alpha_min", 0.05)
        self.alpha_max = config.get("alpha_max", 0.5)
        self.alpha = self.alpha_min  # start conservative; updated adaptively
        self.beta = config.get("beta", 1.0)

        # PPO hyper-parameters
        self.ppo_epochs = config.get("ppo_epochs", 4)
        self.ppo_clip = config.get("ppo_clip", 0.2)
        self.ppo_lr = config.get("ppo_lr", 3e-4)
        self.gamma_discount = config.get("gamma_discount", 0.99)

        self.epsilons = []
        self.subspace_U = None
        self.ppo_optimizer = None

        self.agent_ids = None
        self.mu_model = None
        self.mu_model_flatten_params = None
        self.es_optimizer = None

        # History of PPO gradient directions for adaptive alpha
        self._grad_dir_history = deque(maxlen=10)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_population(self, policy, env):
        self.agent_ids = env.get_agent_ids()
        self.mu_model = policy
        self.mu_model_flatten_params = torch.tensor(
            get_flatten_params(self.mu_model)['params'],
            requires_grad=True, dtype=torch.float32
        )
        n_params = self.mu_model_flatten_params.shape[0]
        print(f'Number of paras ES will updated: {n_params}')

        self.es_optimizer = torch.optim.Adam(
            [self.mu_model_flatten_params], lr=self.learning_rate
        )

        # PPO optimizer only for Critic (value_head), not for Actor
        if hasattr(self.mu_model, 'model') and hasattr(self.mu_model.model, 'value_head'):
            self.ppo_optimizer = torch.optim.Adam(
                self.mu_model.model.value_head.parameters(), lr=self.ppo_lr
            )

        k = min(self.subspace_dim, n_params)
        self.subspace_U, _ = np.linalg.qr(np.random.randn(n_params, k))

        return self.init_perturbations(self.agent_ids, self.mu_model, self.sigma_curr, self.population_size)

    # ------------------------------------------------------------------
    # Perturbation generation
    # ------------------------------------------------------------------

    def init_perturbations(self, agent_ids, mu_model, sigma, pop_size):
        """
        Generate guided perturbations:
            ε_i = α * (U @ z1) + β * z2
        where z1 ~ N(0, I_k), z2 ~ N(0, I_n), U is the (n, k) subspace matrix.
        """
        perturbations = []
        self.epsilons = []

        # Parent (un-perturbed) policy is always the first individual
        perturbations.append(agent_policy(agent_ids, mu_model))
        self.epsilons.append(np.zeros(self.mu_model_flatten_params.shape[0]))

        mu_params = get_flatten_params(mu_model)['params']
        n = mu_params.shape[0]
        k = self.subspace_U.shape[1]

        for _num in range(pop_size):
            perturbed_policy = deepcopy(mu_model)
            perturbed_policy.set_policy_id(_num)

            z1 = np.random.randn(k)
            z2 = np.random.randn(n)
            epsilon = self.alpha * (self.subspace_U @ z1) + self.beta * z2

            perturbed_params = mu_params + epsilon * sigma
            lengths = get_flatten_params(perturbed_policy)['lengths']
            set_flatten_params(perturbed_params, lengths, perturbed_policy)

            perturbations.append(agent_policy(agent_ids, perturbed_policy))
            self.epsilons.append(epsilon)

        return perturbations

    # ------------------------------------------------------------------
    # Adaptive alpha
    # ------------------------------------------------------------------

    def update_alpha(self, grad_ppo):
        """Update alpha based on consistency of recent PPO gradient directions."""
        norm = np.linalg.norm(grad_ppo)
        if norm < 1e-8:
            return
        grad_dir = grad_ppo / norm
        self._grad_dir_history.append(grad_dir)

        if len(self._grad_dir_history) < 2:
            return

        # Mean cosine similarity of current direction vs all history
        sims = [float(np.dot(grad_dir, h)) for h in list(self._grad_dir_history)[:-1]]
        grad_consistency = float(np.mean(sims))  # ∈ [-1, 1]

        consistency_normalized = (grad_consistency + 1.0) / 2.0  # → [0, 1]
        self.alpha = self.alpha_min + consistency_normalized * (self.alpha_max - self.alpha_min)

    # ------------------------------------------------------------------
    # Change-aware subspace update
    # ------------------------------------------------------------------

    def update_subspace(self, grad_ppo, delta_theta_norm):
        """
        Roll-refresh the subspace columns proportionally to the ES parameter change magnitude.
        First column gets the exact new PPO direction; subsequent refreshed columns add small
        random noise to avoid column degeneracy. QR orthogonalization is applied afterwards.
        """
        norm = np.linalg.norm(grad_ppo)
        if norm < 1e-8:
            return
        new_dir = grad_ppo / norm

        n_params = self.subspace_U.shape[0]
        k = self.subspace_U.shape[1]

        # How many columns to refresh this step
        baseline = self.learning_rate * np.sqrt(n_params)
        relative_change = delta_theta_norm / (baseline + 1e-8)
        refresh_cols = int(np.clip(relative_change, 1, k // 2))

        # Roll so the "oldest" column is last; replace last `refresh_cols` columns
        self.subspace_U = np.roll(self.subspace_U, shift=-refresh_cols, axis=1)
        self.subspace_U[:, -1] = new_dir
        for i in range(1, refresh_cols):
            noise = np.random.randn(n_params) * 0.01
            self.subspace_U[:, -(i + 1)] = new_dir + noise

        self.subspace_U, _ = np.linalg.qr(self.subspace_U)

    # ------------------------------------------------------------------
    # Subspace alignment monitoring
    # ------------------------------------------------------------------

    def compute_subspace_alignment(self, grad_ppo_current):
        """
        Measure how well the current PPO gradient direction is captured by the subspace U.
        Returns alignment ∈ [0, 1].
        """
        norm = np.linalg.norm(grad_ppo_current)
        if norm < 1e-8:
            return 0.0
        grad_dir = grad_ppo_current / norm
        projection = self.subspace_U @ (self.subspace_U.T @ grad_dir)
        alignment = float(np.linalg.norm(projection))
        return alignment

    # ------------------------------------------------------------------
    # PPO surrogate gradient
    # ------------------------------------------------------------------

    def compute_ppo_surrogate_gradient(self, trajectories, device):
        """
        Compute the PPO surrogate gradient for the Actor using collected trajectories.

        Reward handling: only the last step reward is non-zero (-VMcost - SLApenalty).
        The return G_t = gamma^(T-t) * R_total is computed via standard discounting.

        Only Actor parameters (excluding value_head) contribute to the returned gradient.
        The Critic (value_head) is separately updated via self.ppo_optimizer.

        Returns:
            numpy array of shape (n_params,) — mean Actor gradient across ppo_epochs,
            or None if trajectories is empty.
        """
        if not trajectories:
            return None

        actions = torch.tensor([t['action'] for t in trajectories], dtype=torch.long)
        log_probs_old = torch.tensor(
            [t['log_prob'] for t in trajectories], dtype=torch.float32
        )
        rewards_list = [t['reward'] for t in trajectories]

        # Discount returns from sparse terminal reward
        returns = []
        G = 0.0
        for r in reversed(rewards_list):
            G = r + self.gamma_discount * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        actor_grad_accum = None

        for epoch in range(self.ppo_epochs):
            log_probs_new = []
            values_new = []

            for t in trajectories:
                logits, value = self.mu_model.model(
                    device, t['ob'], t['dag'], t['node_id'], t['VM_features_matrix']
                )
                logits = logits.squeeze()
                dist = torch.distributions.Categorical(logits=logits)
                log_probs_new.append(dist.log_prob(torch.tensor(t['action'])))
                values_new.append(value.squeeze())

            log_probs_new = torch.stack(log_probs_new)
            values_new = torch.stack(values_new)

            # Advantage: normalize for stability
            advantages = returns - values_new.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO clip loss (Actor)
            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages.detach()
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic MSE loss
            critic_loss = nn.MSELoss()(values_new, returns.detach())

            # Actor backward — collect gradient, but do NOT step ES optimizer
            self.mu_model.zero_grad()
            actor_loss.backward(retain_graph=True)

            grad_vec = torch.cat([
                p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                for name, p in self.mu_model.named_parameters()
                if 'value_head' not in name  # exclude Critic
            ]).detach().numpy()

            actor_grad_accum = grad_vec if actor_grad_accum is None else actor_grad_accum + grad_vec

            # Critic step (separate optimizer, only value_head parameters)
            if self.ppo_optimizer is not None:
                self.ppo_optimizer.zero_grad()
                critic_loss.backward()
                self.ppo_optimizer.step()

        return actor_grad_accum / self.ppo_epochs

    # ------------------------------------------------------------------
    # Main ES update step
    # ------------------------------------------------------------------

    def next_population(self, assemble, results, g, trajectories=None, device='cpu'):
        """
        1. Compute ES gradient estimate from population rewards.
        2. Step ES optimizer (updates Actor parameters).
        3. If trajectories provided: compute PPO gradient → update alpha, subspace.
           Otherwise fall back to using ES gradient for subspace update.
        4. Generate and return the next population.
        """
        rewards = results['rewards'].tolist()
        best_reward_per_g = max(rewards)
        rewards = np.array(rewards)

        if self.reward_shaping:
            rewards = compute_centered_ranks(rewards)
        if self.reward_norm:
            r_std = rewards.std()
            if r_std > 1e-8:
                rewards = (rewards - rewards.mean()) / r_std

        # ES gradient estimate (same formula as es_openai.py)
        update_factor = -1.0 / ((len(self.epsilons) - 1) * self.sigma_curr)
        grad_es = np.sum(
            np.array(self.epsilons) * rewards.reshape(-1, 1), axis=0
        ) * update_factor

        # Record parameters before ES step
        theta_before = self.mu_model_flatten_params.clone().detach().numpy().copy()

        # ES parameter update (Actor only via flat param vector)
        self.mu_model_flatten_params.grad = torch.tensor(grad_es, dtype=torch.float32)
        self.es_optimizer.step()
        flatten_params = self.mu_model_flatten_params.clone()
        set_flatten_params(
            flatten_params.detach().numpy(),
            get_flatten_params(self.mu_model)['lengths'],
            self.mu_model
        )

        # Measure how much ES changed the parameters this generation
        theta_after = self.mu_model_flatten_params.detach().numpy()
        delta_theta_norm = float(np.linalg.norm(theta_after - theta_before))

        # Subspace update: prefer PPO gradient direction, fall back to ES gradient
        if trajectories:
            grad_ppo = self.compute_ppo_surrogate_gradient(trajectories, device)
            if grad_ppo is not None:
                self.update_alpha(grad_ppo)
                alignment = self.compute_subspace_alignment(grad_ppo)
                self.update_subspace(grad_ppo, delta_theta_norm)
                print(
                    f"[GuidedES-PPO] alpha={self.alpha:.4f}, "
                    f"alignment={alignment:.4f}, "
                    f"delta_theta={delta_theta_norm:.6f}",
                    flush=True
                )
            else:
                self.update_subspace(grad_es, delta_theta_norm)
        else:
            self.update_subspace(grad_es, delta_theta_norm)

        perturbations = self.init_perturbations(
            self.agent_ids, self.mu_model, self.sigma_curr, self.population_size
        )

        if self.sigma_curr >= 0.01:
            self.sigma_curr *= self.sigma_decay

        return perturbations, self.sigma_curr, best_reward_per_g

    def get_elite_model(self):
        return self.mu_model


# ------------------------------------------------------------------
# Helpers (same as es_openai.py)
# ------------------------------------------------------------------

def compute_ranks(x):
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= 0.5
    return y
