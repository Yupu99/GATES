"""
Guided ES update policy
- Ensures subspace and flat param dimensions always match
- Uses a small surrogate REINFORCE step to update guiding subspace
"""
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from optim.base_optim import BaseOptim
from utils.policy_dict import agent_policy
from utils.torch_util import get_flatten_params, set_flatten_params

# SubspaceManager 
class SubspaceManager:
    """
    Maintain an orthonormal guiding subspace U (n x k) built from the
    most recent surrogate gradients (FIFO buffer).
    """
    
    def __init__(self, k: int, n: int):
        assert k >= 1 and n >= 1

        # k: The dimension of the subspace (e.g., 3 or 5). Small k = high bias.
        self.k = k
         # n: The dimension of the full parameter space
        self.n = n
        # buffer: A list to store the last k surrogate gradient vectors.
        self.buffer = []
        # U: The calculated Orthonormal Basis matrix (shape: n x current_k).
        self.U = None

    def _reinit(self, n_new: int):
        """Clear buffer and reset if dimension changed."""
        self.n = n_new
        self.buffer = []
        self.U = None

    def update(self, new_grad: np.ndarray):
        """
        Adds a new gradient vector (1D numpy array) to the FIFO buffer and recompute orthonormal basis U.
        Skips near-zero grads to avoid numerical issues.
        """
        if new_grad.ndim != 1:
            raise ValueError("new_grad must be a 1D numpy array")
        if new_grad.shape[0] != self.n:
            # dimension mismatch: reinit subspace
            self._reinit(new_grad.shape[0])
        
        # skip near-zero updates (prevent QR degeneracy)
        if float(np.linalg.norm(new_grad)) < 1e-12:
            return

        # Append and enforce FIFO: If buffer is full, remove the oldest gradient
        self.buffer.append(new_grad.copy())
        if len(self.buffer) > self.k:
            self.buffer.pop(0)

        # build matrix (n x cur_k) with columns = buffered grads
        if len(self.buffer) == 0:
            self.U = None
            return

        # Form the Matrix: Stack gradients as columns (shape: n x current_buffer_size).
        # We perform the transpose (.T) so gradients are columns, not rows.
        G = np.vstack(self.buffer).T 

        # Orthogonalize: Perform QR Decomposition.
        # This turns raw correlated gradients into perfect orthogonal directions.
        # mode='reduced' ensures we get a (n x k) matrix, not a massive (n x n).
        Q, _ = np.linalg.qr(G, mode='reduced')

        # Store Q as our guiding subspace U
        self.U = Q  # shape (n, cur_k)

    def get_subspace(self):
        """Returns the current orthogonal basis matrix U."""
        return self.U  # may be None


# ---------- GuidedES class ----------
class GuidedES(BaseOptim):
    """
    Guided ES optimizer for GATES:
    - OpenAI ES style gradient estimator
    - guided epsilon sampling using a learned subspace U
    - surrogate gradient is used only to shape epsilon sampling
    """

    def __init__(self, config):
        super(GuidedES, self).__init__()
        self.name = config["name"]

        # ES params
        self.sigma_init = float(config["sigma_init"])
        self.sigma_curr = float(self.sigma_init)
        self.sigma_decay = float(config["sigma_decay"])
        self.learning_rate = float(config["learning_rate"])

        # Guided ES params
        # k : Dimension of subspace (e.g., 3)
        self.subspace_k = int(config.get("subspace_k", 3))
        # alpha: the mixing coefficient 
        # (0.5 = 50% guide, 50% random - recommended by Maheswaranathan's paper)
        self.guided_alpha = float(config.get("guided_alpha", 0.5))
        self.surrogate_episodes = int(config.get("surrogate_episodes", 2))
        self.reinforce_learning_rate = float(config.get("reinforce_learning_rate", 0.0))

        # Population + reward handling
        self.population_size = int(config["population_size"])
        self.reward_shaping = bool(config.get('reward_shaping', False))
        self.reward_norm = bool(config.get('reward_norm', False))

        self.maximization = bool(config.get("maximization", True))

        # State
        self.epsilons = []                # save epsilons with respect to every model
        self.agent_ids = None
        self.mu_model = None
        self.mu_model_flat = None        
        self.optimizer = None
        self.subspace = None              # Initialise the Subspace Manager (n is unknown until init_population)
        self.run_seed = None              # keep track of the run seed to build stable surrogate reset seeds
        self.num_train_instances = None   # how many train instances exist 

    # --- helpers for flattening ---
    def _model_flat_numpy(self, model: nn.Module) -> np.ndarray:
        """Return numpy 1D array of flattened parameters in the same order as get_flatten_params."""
        # We rely on get_flatten_params utility to provide lengths, but to be robust
        # we construct by iterating model.parameters() which should match get_flatten_params ordering.
        parts = []
        for p in model.parameters():
            parts.append(p.data.detach().cpu().numpy().ravel())
        if len(parts) == 0:
            return np.array([], dtype=np.float64)
        return np.concatenate(parts).astype(np.float64)

    def _grads_flat_numpy(self, model: nn.Module):
        """Flatten gradients from model.parameters() into a single 1D numpy array.
           If some param has no grad, treat it as zeros.
        """
        parts = []
        for p in model.parameters():
            if p.grad is None:
                parts.append(np.zeros(p.numel(), dtype=np.float64))
            else:
                parts.append(p.grad.detach().cpu().numpy().ravel().astype(np.float64))
        if len(parts) == 0:
            return np.array([], dtype=np.float64)
        return np.concatenate(parts)

    def _ensure_subspace_matches(self, n_params: int):
        """If subspace is None or has wrong n, (re)create it."""
        if self.subspace is None:
            self.subspace = SubspaceManager(k=self.subspace_k, n=n_params)
        elif self.subspace.n != n_params:
            # reinit with new dimension
            self.subspace = SubspaceManager(k=self.subspace_k, n=n_params)

    def _infer_num_train_instances(self, env) -> int:
        for attr in ["evalNum"]:
            if hasattr(env, attr):
                try:
                    v = int(getattr(env, attr))
                    if v > 0:
                        return v
                except Exception:
                    pass
        return 10

    def _make_surrogate_reset(self, env, g: int, ep: int):
        """
        Decide:
        - ep_num: which training instance to use (varies across surrogate episodes)
        - reset_seed: depends on run_seed + generation + ep_num
        """
        if self.num_train_instances is None:
            self.num_train_instances = self._infer_num_train_instances(env)

        ep_num = (int(g) + int(ep)) % int(self.num_train_instances)

        data_gen = None
        if hasattr(env, "set") and hasattr(env.set, "dataGen"):
            try:
                data_gen = int(env.set.dataGen)
            except Exception:
                data_gen = None

        if data_gen is not None and data_gen >= 0:
            gen_idx = int(g) % (data_gen + 1)   
        else:
            gen_idx = int(g)  

        base = int(self.run_seed) if self.run_seed is not None else 42
        rng_seed = base * 100000 + int(gen_idx) * 100 + int(ep_num)

        return gen_idx, ep_num, rng_seed
    
    def init_population(self, policy: torch.nn.Module, env):
        """
        Called once at startup:
        - Set center policy
        - create flat parameter container + optimizer
        - initialize subspace manager
        - sample first population
        """
        import os

        # --- Determine run seed robustly ---
        run_seed = None

        # 1) Try env.seed if available
        if hasattr(env, "seed"):
            try:
                run_seed = int(getattr(env, "seed"))
            except Exception:
                run_seed = None

        # 2) Fall back to SLURM-provided env var
        if run_seed is None:
            env_seed = os.getenv("GATES_SEED")
            if env_seed is not None:
                try:
                    run_seed = int(env_seed)
                except ValueError:
                    run_seed = None

        # 3) Final fallback
        if run_seed is None:
            run_seed = 42

        self.run_seed = run_seed

        # --- Normal init ---
        self.agent_ids = env.get_agent_ids()
        self.num_train_instances = self._infer_num_train_instances(env)

        # Initialize parameters of the policy
        policy.norm_init()
        self.mu_model = policy

        # Build a torch parameter that represents flattened params for optimization
        flat_np = self._model_flat_numpy(self.mu_model)
        self.mu_model_flat = torch.nn.Parameter(
            torch.tensor(flat_np, dtype=torch.float64),
            requires_grad=True
        )

        # Adam optimizer on the flattened parameter vector
        self.optimizer = torch.optim.Adam([self.mu_model_flat], lr=self.learning_rate)

        # Ensure subspace manager knows the dimension
        n_params = flat_np.shape[0]
        self._ensure_subspace_matches(n_params)

        # Guided ES samples populations per-generation in prepare_population
        return None

    def sample_guided_epsilon(self, dim: int) -> np.ndarray:
        """
        Returns epsilon ~ N(0, Cov) where Cov is chosen so that:
        - when no subspace exists: epsilon ~ N(0, I)  (OpenAI ES baseline scale)
        - when subspace exists: epsilon has trace ~ n (so sigma means same thing as vanilla ES)

        Implementation:
            eps = sqrt(alpha) * z + sqrt(1-alpha) * sqrt(n/k) * (U @ v)
            where z ~ N(0, I_n), v ~ N(0, I_k)
        This keeps epsilon scale comparable to vanilla ES but biases directions into U.
        """

        n = int(dim)
        alpha = float(self.guided_alpha)
        U = self.subspace.get_subspace() if self.subspace is not None else None

        # If no guide (e.g., first step before any update), we use simple isotropic Gaussian
        if U is None:
            return np.random.normal(size=(n,)).astype(np.float64)

        # current_k is actual number of columns in U (may be <= self.subspace_k during warmup)
        current_k = U.shape[1]
        if current_k == 0:
            return np.random.normal(size=(n,)).astype(np.float64)

        z = np.random.normal(size=(n,)).astype(np.float64)
        v = np.random.normal(size=(current_k,)).astype(np.float64)

        guided = U @ v  # shape (n,)

        eps = (np.sqrt(alpha) * z) + (np.sqrt(1.0 - alpha) * np.sqrt(n / current_k) * guided)
        return eps.astype(np.float64)

    def init_perturbations(self, agent_ids: list, mu_model: torch.nn.Module, sigma: float, pop_size: int):
        """
        Population layout:
        index 0: center
        indices 1..pop_size: offspring
        """
        perturbations = []
        self.epsilons = []

        # center
        perturbations.append(agent_policy(agent_ids, mu_model))
        self.epsilons.append(np.zeros(int(self.mu_model_flat.numel()), dtype=np.float64))

        # dim + ensure subspace
        n_params = int(self.mu_model_flat.numel())
        self._ensure_subspace_matches(n_params)

        # authoritative center vector for all offspring
        base_flat = self.mu_model_flat.detach().cpu().numpy().astype(np.float64)

        for i in range(int(pop_size)):
            pert = deepcopy(mu_model)
            pert.set_policy_id(i)

            eps = self.sample_guided_epsilon(n_params)
            perturbed_flat = base_flat + (float(sigma) * eps)

            # write into pert
            lengths = get_flatten_params(pert)["lengths"]
            set_flatten_params(perturbed_flat, lengths, pert)

            perturbations.append(agent_policy(agent_ids, pert))
            self.epsilons.append(eps)

        return perturbations

    def compute_surrogate_gradient(self, env, g):
        """
        REINFORCE-style surrogate gradient at center policy.
        This serves as the 'Guide' for the noise generation.
        Used ONLY to build guiding subspace (directional signal).
        """

        n = int(self.mu_model_flat.numel())

        if self.reinforce_learning_rate <= 0.0 or self.surrogate_episodes <= 0:
            return np.zeros(n, dtype=np.float64)

        # Ensure we know num_train_instances
        if self.num_train_instances is None:
            self.num_train_instances = self._infer_num_train_instances(env)

        # Put model into training mode so gradients can flow
        self.mu_model.train()

        # compute gradients w.r.t model parameters (not the flat parameter), then flatten
        # Zero grads
        for p in self.mu_model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        # Number of episodes to run for the surrogate (keep small for speed, e.g., 2-4)
        num_eps = int(self.surrogate_episodes)
    
        logprob_sums = []
        returns = []

        for ep in range(num_eps):

            gen_idx, ep_num, rng_seed = self._make_surrogate_reset(env, g=g, ep=ep)

            # Seed randomness for the surrogate rollout 
            np.random.seed(int(rng_seed))
            torch.manual_seed(int(rng_seed))

            state_dict_list = env.reset(gen_idx, ep_num, "train")

            agent_data = state_dict_list["0"]
            ob = agent_data["state"]
            dag = agent_data["DAG"]
            node_id = agent_data["Node_id"]
            done = False

            log_probs = []
            rewards = []

            while not done:
                # Forward Pass : Pass unpacked data to the policy
                remove_vm_idx = state_dict_list.get("removeVM", None)
                # Pass Graph Data (DAG) and Node_id
                action, log_prob = self.mu_model.get_action_and_log_prob(ob, dag, node_id, removeVM=remove_vm_idx)
                # Step environment
                state_dict_list, r, done, info = env.step({"0": action})
                log_probs.append(log_prob)
                # Reward Handling
                # GATES defines reward = -(Cost + Penalty). 
                rewards.append(r)

                # Update 'ob' for the next iteration
                if not done:
                    agent_data = state_dict_list["0"]
                    ob = agent_data["state"]
                    dag = agent_data["DAG"]
                    node_id = agent_data["Node_id"]
        
            # Compute REINFORCE Loss for this episode
            # Loss = - sum(log_prob * Return)
            R = float(np.sum(rewards))
            returns.append(R)

            if len(log_probs) == 0:
                logprob_sums.append(None)
            else:
                logprob_sums.append(torch.stack(log_probs).sum())

        # baseline to reduce variance (simple mean baseline)
        baseline = float(np.mean(returns)) if len(returns) > 0 else 0.0

        total_loss = 0.0
        count = 0
        for lp_sum, R in zip(logprob_sums, returns):
            if lp_sum is None:
                continue
            # REINFORCE loss: -logpi * (R - b)
            total_loss = total_loss + (-lp_sum * float(R - baseline))
            count += 1

        if count == 0:
            return np.zeros(n, dtype=np.float64)

        total_loss = total_loss / float(count)

        # Backpropagate to populate .grad for model params
        total_loss.backward()

        # Flatten parameter grads (same param order as get_flatten_params)
        flat_grad = self._grads_flat_numpy(self.mu_model)

        # NaN/Inf guard
        if not np.all(np.isfinite(flat_grad)):
            # clear grads
            for p in self.mu_model.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            return np.zeros(n, dtype=np.float64)

        # clear grads to be safe
        for p in self.mu_model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        return flat_grad.astype(np.float64)

    def prepare_population(self, env, g: int):
        """
        Guided ES ordering:
        1) compute surrogate gradient
        2) update subspace
        3) sample guided perturbations for this generation
        """
        surrogate_grad = self.compute_surrogate_gradient(env, g)
        self._ensure_subspace_matches(surrogate_grad.shape[0])
        self.subspace.update(surrogate_grad)

        if self.run_seed is not None:
            np.random.seed(int(self.run_seed) * 100000 + int(g))

        return self.init_perturbations(self.agent_ids, self.mu_model, self.sigma_curr, self.population_size)

    # Guided ES Next Population
    def next_population(self, assemble, results, g: int, env):
        """
        Performs standard ES update on the flattened parameter vector using stored epsilons and rewards.
        """
        rewards_list = results['rewards'].tolist()
        rewards = np.asarray(rewards_list, dtype=np.float64)

        best_reward_per_g = float(np.max(rewards))

        # fitness shaping
        if self.reward_shaping:
            rewards = compute_centered_ranks(rewards.astype(np.float32)).astype(np.float64)

        # reward normalization 
        if self.reward_norm:
            r_std = float(rewards.std())
            if r_std > 1e-12:
                rewards = (rewards - rewards.mean()) / r_std
            else:
                rewards = rewards - rewards.mean()

        # offspring only (index 0 is center)
        eps_off = np.stack(self.epsilons[1:]).astype(np.float64)            # (pop, n)
        rew_off = rewards[1:].astype(np.float64)                             # (pop,)

        # ES gradient estimate (OpenAI ES):
        # grad ≈ (1/(N*sigma)) * sum_i rew_i * eps_i
        # baseline uses a negative sign for minimization-style update.
        # make it explicit:
        if self.maximization:
            # maximize rewards: Adam does gradient descent, so negate ascent direction
            sign = -1.0
        else:
            # minimize rewards: standard descent direction
            sign = +1.0

        update_factor = sign / (float(self.population_size) * float(self.sigma_curr) + 1e-12)
        grad_est = np.sum(eps_off * rew_off[:, None], axis=0) * update_factor  # (n,)

        # write grad into flat parameter and step
        self.mu_model_flat.grad = torch.tensor(grad_est, dtype=torch.float64)
        self.optimizer.step()

        # sync mu_model weights from mu_model_flat
        flat_after = self.mu_model_flat.detach().cpu().numpy().astype(np.float64)
        lengths = get_flatten_params(self.mu_model)["lengths"]
        set_flatten_params(flat_after, lengths, self.mu_model)

        if self.sigma_curr >= 0.01:
            self.sigma_curr *= self.sigma_decay

        return None, self.sigma_curr, best_reward_per_g

    def get_elite_model(self):
        return self.mu_model

def compute_ranks(x):
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks

def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y
