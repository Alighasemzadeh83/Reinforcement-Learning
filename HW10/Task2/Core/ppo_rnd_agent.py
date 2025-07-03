# File: Core/ppo_rnd_agent.py

import torch
import numpy as np
from torch.optim.adam import Adam
from Core.model import PolicyModel, PredictorModel, TargetModel
from Common.utils import mean_of_list, RunningMeanStd
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True  # Optional performance boost


class Brain:
    def __init__(self, **config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Models ---
        self.current_policy = PolicyModel(config["state_shape"], config["n_actions"]).to(self.device)
        self.predictor_model = PredictorModel(config["obs_shape"]).to(self.device)
        self.target_model = TargetModel(config["obs_shape"]).to(self.device)
        for param in self.target_model.parameters():
            param.requires_grad = False  # Keep target fixed

        # --- Optimizer: policy + predictor only (target is frozen) ---
        self.optimizer = Adam(
            list(self.current_policy.parameters()) + list(self.predictor_model.parameters()),
            lr=config["lr"]
        )

        # --- Normalization buffers ---
        self.state_rms = RunningMeanStd(shape=config["obs_shape"])
        self.int_reward_rms = RunningMeanStd(shape=(1,))
        self.mse_loss = torch.nn.MSELoss(reduction='none')  # we will handle masking

    def get_actions_and_values(self, obs_tensor, hidden_state):
        obs_tensor = obs_tensor.to(self.device)
        hidden_state = hidden_state.to(self.device)
        with torch.no_grad():
            dist, int_val, ext_val, probs, new_hidden = self.current_policy(obs_tensor, hidden_state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.cpu(), int_val.cpu(), ext_val.cpu(), log_prob.cpu(), probs.cpu(), new_hidden.cpu()

    def calculate_int_rewards(self, next_obs, batch=True):
        """
        next_obs: if batch=False → shape [C, H, W], values [0,255]
                  if batch=True  → shape [B, C, H, W]
        Returns: torch.Tensor (B,) of intrinsic rewards (MSE)
        """
        if not batch:
            next_obs = np.expand_dims(next_obs, axis=0)  # → [1, C, H, W]

        # 1) Normalize by running Rms:
        norm_obs = np.clip(
            (next_obs - self.state_rms.mean) / (np.sqrt(self.state_rms.var) + 1e-8),
            -5, 5
        ).astype(np.float32)

        norm_obs = torch.tensor(norm_obs, dtype=torch.float32).to(self.device)  # [B, C, H, W]

        # 2) Get target + predictor features
        with torch.no_grad():
            target_feat = self.target_model(norm_obs)       # [B, 512]
        pred_feat = self.predictor_model(norm_obs)           # [B, 512]

        # 3) MSE per sample: (pred - target)**2 → mean over 512 dims
        se = (pred_feat - target_feat).pow(2)                # [B, 512]
        mse_per_sample = se.mean(dim=1)                      # [B]

        if not batch:
            return mse_per_sample.cpu().squeeze().detach()   # scalar tensor
        else:
            return mse_per_sample.cpu().detach().numpy()     # numpy array of shape [B]

    def normalize_int_rewards(self, int_rewards):
        """
        int_rewards: list of lists of floats (for each rollout)
        Returns: same-shaped list of normalized intrinsic rewards
        """
        gamma = self.config["int_gamma"]
        returns = []
        for rewards in int_rewards:
            discounted, acc = [], 0
            for r in reversed(rewards):
                acc = r + gamma * acc
                discounted.insert(0, acc)
            returns.append(discounted)

        flat = np.ravel(returns).reshape(-1, 1)
        self.int_reward_rms.update(flat)

        return int_rewards / (np.sqrt(self.int_reward_rms.var) + 1e-8)

    def get_gae(self, rewards, values, next_values, dones, gamma):
        lam = self.config["lambda"]
        advantages = []

        for r, v, nv, d in zip(rewards, values, next_values, dones):
            adv, gae = [], 0
            for t in reversed(range(len(r))):
                delta = r[t] + gamma * nv[t] * (1 - d[t]) - v[t]
                gae = delta + gamma * lam * (1 - d[t]) * gae
                adv.insert(0, gae)
            advantages.append(adv)

        return np.array(advantages)

    @mean_of_list
    def train(self, states, actions, int_rewards, ext_rewards, dones,
              int_values, ext_values, log_probs, next_int_values,
              next_ext_values, total_next_obs, hidden_states):

        # --- Advantage Calculation ---
        int_returns = self.get_gae(
            [int_rewards], [int_values], [next_int_values],
            [np.zeros_like(dones)], self.config["int_gamma"]
        )[0]
        ext_returns = self.get_gae(
            [ext_rewards], [ext_values], [next_ext_values],
            [dones], self.config["ext_gamma"]
        )[0]

        advs = (ext_returns - ext_values) * self.config["ext_adv_coeff"] + \
               (int_returns - int_values) * self.config["int_adv_coeff"]

        advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
        ext_returns = torch.tensor(ext_returns, dtype=torch.float32, device=self.device)
        int_returns = torch.tensor(int_returns, dtype=torch.float32, device=self.device)

        # --- Prepare inputs ---
        states = states.to(self.device)                    # [T×B, C, H, W]
        actions = actions.to(self.device)                  # [T×B]
        log_probs = log_probs.to(self.device)              # [T×B]
        hidden_states = hidden_states.to(self.device)      # [T×B, hidden_dim]
        next_obs = torch.tensor(total_next_obs, dtype=torch.float32).to(self.device)
        # total_next_obs is numpy [T×B, C, H, W]

        pg_losses, ext_v_losses, int_v_losses, rnd_losses, entropies = [], [], [], [], []

        for _ in range(self.config["n_epochs"]):
            # 1) Re‐evaluate current policy on the same states
            dist, int_val, ext_val, _, _ = self.current_policy(states, hidden_states)
            entropy = dist.entropy().mean()
            new_log_prob = dist.log_prob(actions)
            ratio = (new_log_prob - log_probs).exp()

            # 2) PPO Surrogate Loss
            surr1 = ratio * advs
            surr2 = torch.clamp(ratio,
                                1 - self.config["clip_range"],
                                1 + self.config["clip_range"]) * advs
            pg_loss = -torch.min(surr1, surr2).mean()

            # 3) Value Function Losses
            v_ext_loss = self.mse_loss(ext_val.squeeze(), ext_returns).mean()
            v_int_loss = self.mse_loss(int_val.squeeze(), int_returns).mean()
            critic_loss = 0.5 * (v_ext_loss + v_int_loss)

            # --- RND Loss ---
            rnd_loss = self.calculate_rnd_loss(next_obs)

            # 4) Total Loss
            loss = pg_loss + critic_loss + rnd_loss - self.config["ent_coeff"] * entropy

            # 5) Backprop & Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.current_policy.parameters(),
                                          self.config["max_grad_norm"])
            self.optimizer.step()

            # 6) Record per‐epoch metrics
            pg_losses.append(pg_loss.item())
            ext_v_losses.append(v_ext_loss.item())
            int_v_losses.append(v_int_loss.item())
            rnd_losses.append(rnd_loss.item())
            entropies.append(entropy.item())

        # --- Return exactly 9 lists, so mean_of_list can compute explained_variance on the last two pairs ---
        return [
            pg_losses,
            ext_v_losses,
            int_v_losses,
            rnd_losses,
            entropies,
            np.array(int_values, dtype=np.float32),
            np.array(int_returns.cpu().numpy(), dtype=np.float32),
            np.array(ext_values, dtype=np.float32),
            np.array(ext_returns.cpu().numpy(), dtype=np.float32)
        ]

    def calculate_rnd_loss(self, obs):
        """
        obs: torch.Tensor [B, C, H, W], values in [0,255]
        Returns: scalar RND loss = masked MSE(pred, target)
        """
        # 1) Normalize by state_rms (borrowed from calculate_int_rewards)
        with torch.no_grad():
            obs_np = obs.cpu().numpy()  # [B, C, H, W]
        norm_obs = np.clip(
            (obs_np - self.state_rms.mean) / (np.sqrt(self.state_rms.var) + 1e-8),
            -5, 5
        ).astype(np.float32)
        norm_obs = torch.tensor(norm_obs, dtype=torch.float32).to(self.device)

        # 2) Forward through target (frozen) and predictor
        with torch.no_grad():
            target_feat = self.target_model(norm_obs)  # [B, 512]
        pred_feat = self.predictor_model(norm_obs)    # [B, 512]

        # 3) Squared error per-element
        se = (pred_feat - target_feat).pow(2)         # [B, 512]

        # 4) Create a Bernoulli mask of shape [B, 512]
        B, D = se.shape
        p = self.config["predictor_proportion"]       # e.g. 0.25
        mask = (torch.rand(B, D, device=self.device) < p).float()

        # 5) Zero out some dimensions, then average only over kept dims
        masked_se = se * mask
        keep_counts = mask.sum(dim=1).clamp(min=1.0)  # [B]
        per_sample_loss = masked_se.sum(dim=1) / keep_counts  # [B]

        # 6) Return mean over batch
        return per_sample_loss.mean()

    def set_from_checkpoint(self, checkpoint):
        self.current_policy.load_state_dict(checkpoint["current_policy_state_dict"])
        self.predictor_model.load_state_dict(checkpoint["predictor_model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.state_rms.mean = checkpoint["state_rms_mean"]
        self.state_rms.var = checkpoint["state_rms_var"]
        self.state_rms.count = checkpoint["state_rms_count"]
        self.int_reward_rms.mean = checkpoint["int_reward_rms_mean"]
        self.int_reward_rms.var = checkpoint["int_reward_rms_var"]
        self.int_reward_rms.count = checkpoint["int_reward_rms_count"]

    def set_to_eval_mode(self):
        self.current_policy.eval()
