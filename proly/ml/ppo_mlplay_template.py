"""
PPO MLPlay Module

This module implements a Proximal Policy Optimization (PPO) agent for Unity games
using the MLGame3D framework and Unity ML-Agents PPO implementation.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
import csv, os

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Built-in configuration
TRAINING_MODE = True  # Set to False to use pre-trained model
MODEL_SAVE_DIR = f"./models/{time.strftime('%Y%m%d_%H%M%S')}"
MODEL_LOAD_PATH = "./models/-/model_latest.pt"

# PPO hyperparameters
LEARNING_RATE = 0
GAMMA = 0  # Discount factor
GAE_LAMBDA = 0  # GAE parameter
CLIP_RATIO = 0  # PPO clip parameter
VALUE_COEF = 0  # Value loss coefficient
ENTROPY_COEF = 0  # Entropy loss coefficient
MAX_GRAD_NORM = 0  # Gradient clipping
UPDATE_EPOCHS = 0  # Number of iterations per update
BUFFER_SIZE = 0  # Experience buffer size
BATCH_SIZE = 0  # Batch size
UPDATE_FREQUENCY = 0  # Update frequency
SAVE_FREQUENCY = 0  # Save frequency (episodes)


# Model parameters
HIDDEN_SIZE = 128

# Reward weights
REWARD_WEIGHTS = {
    "checkpoint": 0,  # Passing checkpoint
    "progress": 0,    # Moving toward goal
    "health": 0,      # Health change
    "item_pickup": 0, # Picking up items
    "item_use": 0,    # Using items
    "completion": 0   # Completing level
}
class ObservationProcessor:
    def __init__(self):
        self.observation_size = self._calculate_observation_size()
        print(f"[ObservationProcessor] Size: {self.observation_size}")

    def process(self, obs: Dict[str, Any]) -> np.ndarray:
        return np.array(self._flatten(obs), dtype=np.float32)

    def get_size(self) -> int:
        return self.observation_size

    def _flatten(self, obs):
        f = []

        # === Agent 與 Target 的位置 ===
        agent = np.array(obs["agent_position"])
        target = np.array(obs["target_position"])
        heading = target[[0, 2]] - agent[[0, 2]]
        heading_norm = np.linalg.norm(heading)
        heading_unit = heading / (heading_norm + 1e-6)
        f.extend(heading_unit.tolist())  # (2,)

        # === 距離（normalize） ===
        distance = np.linalg.norm(target - agent) / 30.0  # 30 是地圖最大距離
        f.append(distance)  # (1,)

        # === velocity、方向 ===
        velocity = obs.get("agent_velocity", [0, 0])
        forward = obs.get("agent_forward_direction", [0, 0])
        f.extend(velocity)  # (2,)
        f.extend(forward)   # (2,)

        # === cosine similarity ===
        forward = np.array(forward)
        cos_sim = 0.0
        if heading_norm > 1e-6 and np.linalg.norm(forward) > 1e-6:
            cos_sim = np.dot(forward, heading) / (np.linalg.norm(forward) * heading_norm)
        f.append(cos_sim)  # (1,)

        # === 血量 ===
        f.append(float(obs.get("agent_health_normalized", 1.0)))  # (1,)

        # === 最近障礙物方向（map object + terrain）===
        nearest_obs_dir = None
        min_obs_dist = float("inf")

        for obj in obs["nearby_map_objects"]:
            rel = np.array(obj["relative_position"])
            d = np.linalg.norm(rel)
            if d < min_obs_dist and d > 1e-6:
                min_obs_dist = d
                nearest_obs_dir = rel / d

        for row in obs["terrain_grid"]:
            for cell in row:
                ttype = int(cell["terrain_type"])
                if ttype != 0:
                    rel = np.array(cell["relative_position"])
                    d = np.linalg.norm(rel)
                    if d < min_obs_dist and d > 1e-6:
                        min_obs_dist = d
                        nearest_obs_dir = rel / d

        if nearest_obs_dir is None:
            nearest_obs_dir = np.array([0.0, 0.0])
        f.extend(nearest_obs_dir.tolist())  # (2,)

        return f

    def _calculate_observation_size(self) -> int:
        return 2 + 1 + 2 + 2 + 1 + 1 + 2  # heading(2) + dist(1) + vel(2) + dir(2) + cos(1) + hp(1) + nearest_obs_dir(2)

class ActionProcessor:
    def __init__(self):
        self.action_size = 2

    def create_action(self, output: np.ndarray):
        move = output[:2]
        norm = np.linalg.norm(move)
        if norm > 1.0:
            move = move / norm
        return (
            np.array(move, dtype=np.float32),
            np.array([0, 0], dtype=np.int32)  # 不執行 item 行為
        )

    def get_size(self):
        return self.action_size
    
class RewardCalculator:

    def __init__(self):
        self.prev_pos = None
        self.prev_checkpoint = -1
        self.prev_dist = None
        self.angle_weight = 5.0
        self.dist_weight = 5.0

    def reset(self):
        self.prev_pos = None
        self.prev_checkpoint = -1
        self.prev_dist = None

    def calculate(self, obs: Dict[str, Any], base_reward: float, done: bool = False, info: Any = None) -> float:
        r = 0.0

        agent_pos = np.array(obs.get("agent_position", [0, 0, 0]))
        checkpoint_pos = np.array(obs.get("target_position", [0, 0, 0]))
        forward = np.array(obs.get("agent_forward_direction", [0.0, 0.0]))
        velocity = np.array(obs.get("agent_velocity", [0.0, 0.0]))
        current_terrain = int(obs.get("current_terrain_type", 0))

        # === 初始化距離 ===
        if self.prev_pos is None:
            self.prev_pos = agent_pos
            self.prev_dist = np.linalg.norm(checkpoint_pos[[0, 2]] - agent_pos[[0, 2]])
            return 0.0

        # === 移動向量（xz）===
        move_vec = agent_pos - self.prev_pos
        move_vec[1] = 0
        move_len = np.linalg.norm(move_vec)

        # === 距離縮短（xz）===
        cur_dist = np.linalg.norm(checkpoint_pos[[0, 2]] - agent_pos[[0, 2]])
        delta = self.prev_dist - cur_dist
        if delta > 0.1:
            r += delta * self.dist_weight
        elif delta < -0.1:
            r -= abs(delta) * (self.dist_weight + 1.0)
        self.prev_dist = cur_dist

        # === 角度一致性 ===
        heading_vec = checkpoint_pos[[0, 2]] - self.prev_pos[[0, 2]]
        if np.linalg.norm(heading_vec) > 1e-6 and move_len > 1e-6:
            move_unit = move_vec[[0, 2]] / (move_len + 1e-6)
            heading_unit = heading_vec / (np.linalg.norm(heading_vec) + 1e-6)
            cos_sim = np.dot(move_unit, heading_unit)
            if cos_sim > 0.7:
                r += cos_sim * self.angle_weight
            elif cos_sim < -0.5:
                r -= 2.0

        # === Checkpoint 達成 ===
        cur_cp = int(obs.get("last_checkpoint_index", -1))
        if cur_cp > self.prev_checkpoint:
            r += 20.0
            self.prev_checkpoint = cur_cp
            print(f"[reward] ✅ Reached checkpoint {cur_cp}")

        # === 死亡懲罰 ===
        if agent_pos[1] < 1.1:
            r -= 5.0

        # === 障礙物方向懲罰 ===
        nearest_obs_vec = np.array(obs.get("nearest_obstacle_vec", [0.0, 0.0]))
        obs_dist = np.linalg.norm(nearest_obs_vec)
        if obs_dist < 3.0 and move_len > 1e-3:
            obs_unit = nearest_obs_vec / (obs_dist + 1e-6)
            move_unit = move_vec[[0, 2]] / (move_len + 1e-6)
            danger_cos = np.dot(move_unit, obs_unit)
            if danger_cos > 0.8:
                r -= 5.0  # 朝向障礙物
            elif danger_cos < -0.8:
                r += 3.0  # 背離障礙物

        # === 地形懲罰 ===
        if current_terrain in [-1, 1]:  # 水坑或障礙
            r -= 2.0

        self.prev_pos = agent_pos
        return base_reward + r

        # === Debug log ===
        # print(f"[reward_debug] cp={cur_cp}, delta={delta:.2f}, cos_sim={cos_sim:.2f}, total_r={r:.2f}")

    def reset(self):
        self.prev_checkpoint = -1
        self.prev_dist = float("inf")
        self.prev_pos = None
        self.prev_dist = None
        self.prev_checkpoint = -1


class ExperienceBuffer:
    """
    PPO 的 rollout buffer：儲存 (s, a, r, done, logp, value)
    並計算 advantage / return（只適用於純連續動作）
    """
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, obs, action, reward, done, log_prob, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(bool(done))
        self.values.append(value)
        self.log_probs.append(log_prob.view(-1)) 

    def compute_returns_and_advantages(self, gamma=0.99, lam=0.95):
        advantages = []
        returns = []
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(self.rewards))):
            mask = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            gae = delta + gamma * lam * mask * gae
            next_value = self.values[t]
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])

        self.advantages = torch.tensor(advantages, dtype=torch.float32)
        self.returns = torch.tensor(returns, dtype=torch.float32)
        return self.advantages, self.returns

    def __len__(self):
        return len(self.obs)

class PPOModel(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 128):
        super().__init__()
        # Actor 路徑
        self.actor_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.actor_head = nn.Linear(hidden_size, 2)  # 2 維動作向量

        # Critic 路徑
        self.critic_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.critic_head = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        # 初始 actor weight 為 0（防止初期偏移）
        nn.init.zeros_(self.actor_head.weight)
        nn.init.zeros_(self.actor_head.bias)

    def forward(self, x: torch.Tensor):
        # 這只用於 training loop 取 value（不取 action）
        value_feat = self.critic_encoder(x)
        value = self.critic_head(value_feat).squeeze(-1)
        return None, value

    def act_and_eval(self, x: torch.Tensor):
        actor_feat = self.actor_encoder(x)
        logits = self.actor_head(actor_feat)
        value_feat = self.critic_encoder(x)
        value = self.critic_head(value_feat).squeeze(-1)

        mean = torch.tanh(logits)
        std = torch.ones_like(mean) * 0.1
        dist = torch.distributions.Normal(mean, std)

        sampled_action = dist.sample()
        log_prob = dist.log_prob(sampled_action).sum(dim=-1)

        return sampled_action, log_prob, value

    def evaluate(self, x: torch.Tensor, action: torch.Tensor):
        actor_feat = self.actor_encoder(x)
        logits = self.actor_head(actor_feat)
        mean = torch.tanh(logits)
        std = torch.ones_like(mean) * 0.1
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)

        value_feat = self.critic_encoder(x)
        value = self.critic_head(value_feat).squeeze(-1)

        return log_prob, value

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        if os.path.exists(path):
            try:
                self.load_state_dict(torch.load(path, map_location=device))
                print(f"Model loaded from {path}")
                return True
            except RuntimeError as e:
                print(f"Error loading model from {path}: {e}")
                return False
        else:
            print(f"Model file not found: {path}")
            return False

class MLPlay:
    """
    PPOAgent 主流程，整合環境互動、模型推論、訓練與 reward shaping。
    """
    def __init__(self, action_space_info=None):

        self.player_id = 1
        self.name = f"PPO_Player{self.player_id}"

        self.gamma = 0.99
        self.lam = 0.95
        self.batch_size = 256
        self.update_frequency = 64 #1024
        torch.autograd.set_detect_anomaly(True)
        self.lr = 3e-4
        self.training = True
        self.episode_count = 0
        self.save_frequency = 100
        self.save_dir = f"./models/player{self.player_id}"
        self.model_path = os.path.join(self.save_dir, "model_latest.pt")

        self.clip_ratio = 0.1
        self.entropy_coef = 0.05

        self.obs_proc = ObservationProcessor()
        self.act_proc = ActionProcessor()
        self.reward_calc = RewardCalculator()

        self.model = PPOModel(self.obs_proc.get_size()).to(torch.device("cpu"))
        self.model = PPOModel(input_dim=10).to(device)

        self.model = PPOModel(input_dim=self.obs_proc.get_size()).to(torch.device("cpu"))
        self.model_path = os.path.join(self.save_dir, "model_latest.pt")
        pretrain_path = "supervised_model.pth"

        # === 先讀取 PPO 訓練過的完整模型 ===
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
                print(f"[PPO] ✅ Loaded full PPO model from {self.model_path}")
            except Exception as e:
                print(f"[PPO] ⚠️ Failed to load PPO model: {e}")
        # === 若沒有，再載入 supervised 的 actor 部分權重 ===
        elif os.path.exists(pretrain_path):
            try:
                pretrained = torch.load(pretrain_path, map_location="cpu")
                model_dict = self.model.state_dict()
                pretrained_filtered = {k: v for k, v in pretrained.items() if k in model_dict and "critic" not in k}
                model_dict.update(pretrained_filtered)
                self.model.load_state_dict(model_dict)
                print(f"[PPO] ✅ Loaded actor weights from supervised model: {pretrain_path}")
            except Exception as e:
                print(f"[PPO] ⚠️ Failed to load actor weights from supervised: {e}")
        else:
            print(f"[PPO] ⚠️ No model found, starting from scratch.")

        self.buffer = ExperienceBuffer()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.prev_obs = None
        self.prev_action = None
        self.prev_logp = None
        self.prev_value = None
        self.step = 0

        os.makedirs(self.save_dir, exist_ok=True)
        print("[MLPlay] Initialized")

    def reset(self):
        self.prev_obs = None
        self.prev_action = None
        self.prev_logp = None
        self.prev_value = None
        self.episode_count += 1

    def update(self, obs, reward=0.0, done=False, info=None):

        if done == True:
            print(done)
        obs_tensor = torch.tensor(self.obs_proc.process(obs), dtype=torch.float32).unsqueeze(0)
        action_tensor, logp, value = self.model.act_and_eval(obs_tensor)
        raw_action = action_tensor.squeeze(0).detach().cpu().numpy()
        action = self.act_proc.create_action(raw_action)

        if self.prev_obs is None:
            self.prev_obs = obs_tensor
            self.prev_action = torch.tensor(raw_action)
            self.prev_logp = logp.detach()
            self.prev_value = value.detach()
            return action

        # === 不訓練空中/復活階段 ===
        agent_pos = obs.get("agent_position", [0, 0, 0])
        if agent_pos[1] > 2.0:
            self.prev_obs = obs_tensor
            self.prev_action = torch.tensor(raw_action)
            self.prev_logp = logp.detach()
            self.prev_value = value.detach()
            return action

        reward = self.reward_calc.calculate(obs, reward, done, info)

        self.buffer.add(
            self.prev_obs.squeeze(0),
            self.prev_action,
            reward,
            done,
            self.prev_logp,
            self.prev_value.item()
        )

        self.prev_obs = obs_tensor
        self.prev_action = torch.tensor(raw_action)
        self.prev_logp = logp.detach()
        self.prev_value = value.detach()

        self.step += 1

        if self.training and len(self.buffer) >= self.update_frequency:
            print(f"[train] Player{self.player_id} training at step {self.step}")
            self._train()
            self.buffer.clear()
            if self.episode_count % self.save_frequency == 0:
                self._save_model()

        return action
    
    def _train(self):
        try:
            # === Step 1: Advantage / Return ===
            advantages, returns = self.buffer.compute_returns_and_advantages(self.gamma, self.lam)
            adv = advantages.clone().detach()
            ret = returns.clone().detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            adv = torch.clamp(adv, -5.0, 5.0)



            # === Step 2: 整理資料 ===
            obs = torch.stack(self.buffer.obs)
            act = torch.stack(self.buffer.actions)
            old_logp = torch.stack(self.buffer.log_probs).squeeze(-1)

            # === reward 補強處理 ===
            reward_np = np.array(self.buffer.rewards)
            reward_np = np.clip(reward_np, -5.0, 10.0)
            self.buffer.rewards = reward_np.tolist()

            reward_mean = np.mean(self.buffer.rewards)
            reward_std = np.std(self.buffer.rewards)
            # print(f"[reward_dist] mean={reward_mean:.2f}, std={reward_std:.2f}, min={reward_np.min():.2f}, max={reward_np.max():.2f}")

            # === 若 reward collapse，reset 模型 ===
            if reward_mean < -3.0 and self.episode_count > 5:
                print("[reset] reward collapsed, resetting model weights")
                self.model._init_weights()
                self.buffer.clear()
                return

            clip_ratio = 0.1
            entropy_coef = 0.05

            for _ in range(4):
                # === actor forward ===
                actor_feat = self.model.actor_encoder(obs)
                logits = self.model.actor_head(actor_feat)

                # === critic forward ===
                critic_feat = self.model.critic_encoder(obs)
                value = self.model.critic_head(critic_feat).squeeze(-1)

                # === policy distribution ===
                mean = torch.tanh(logits)
                std = torch.ones_like(mean) * 0.1
                dist = torch.distributions.Normal(mean, std)

                logp = dist.log_prob(act[:, :2]).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                entropy = torch.clamp(entropy, max=2.0)

                # === PPO policy loss ===
                ratio = torch.exp(logp - old_logp)
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
                value_loss = F.mse_loss(value, ret)
                loss = policy_loss + 0.1 * value_loss - 0.01 * entropy

                if not torch.isfinite(loss):
                    print("[warn] ⚠️ Loss is NaN. Skipping training step.")
                    self.buffer.clear()
                    return

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # === Logging ===
            log_path = f"log/player{self.player_id}_metrics.csv"
            os.makedirs("log", exist_ok=True)
            write_header = not os.path.exists(log_path)
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["step", "episode", "policy_loss", "value_loss", "entropy", "total_loss", "avg_reward"])
                writer.writerow([
                    self.step,
                    self.episode_count,
                    policy_loss.item(),
                    value_loss.item(),
                    entropy.item(),
                    loss.item(),
                    reward_mean,
                ])

            self.debug_train_sanity_check(obs, act, old_logp, advantages, returns, logp, value, loss)
            
            self.buffer.clear()

        except Exception as e:
            import traceback
            traceback.print_exc()


    def _save_model(self):
        path = os.path.join(self.save_dir, f"model_ep{self.episode_count}.pt")
        torch.save(self.model.state_dict(), path)
        torch.save(self.model.state_dict(), self.model_path)
        # print(f"[Player {self.player_id}] Model saved to {path}")

    def debug_train_sanity_check(self, obs, act, old_logp, adv, ret, logp, value, loss):
        print("\n========== PPO Training Sanity Check ==========")

        # 1. Reward statistics
        reward_mean = np.mean(self.buffer.rewards)
        reward_std = np.std(self.buffer.rewards)
        print(f"[reward] mean={reward_mean:.4f}, std={reward_std:.4f}")

        # 2. Loss components
        print(f"[loss] policy={loss.item():.4f}, value={F.mse_loss(value.squeeze(), ret).item():.4f}, total={loss.item():.4f}")

        # 3. Log prob dimension check
        print(f"[shape] act={act.shape}, old_logp={old_logp.shape}, logp={logp.shape}")

        # 4. Gradient check (before optimizer.step)
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                print(f"[grad] {name}: grad_norm={norm:.6f}")
            else:
                print(f"[grad] {name}: grad is None (possible detach or no contribution)")

        # 5. Actor weight change check (manual delta compare)
        actor_weight = self.model.actor_head.weight
        actor_bias = self.model.actor_head.bias
        delta_weight = actor_weight.grad.norm().item() if actor_weight.grad is not None else 0.0
        delta_bias = actor_bias.grad.norm().item() if actor_bias.grad is not None else 0.0
        print(f"[update] actor weight grad norm: {delta_weight:.6f}, bias grad norm: {delta_bias:.6f}")

        print("===============================================\n")
