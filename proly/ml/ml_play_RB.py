import numpy as np
import json
import os

class MLPlay:
    def __init__(self, action_space_info=None):
        self.name = "RuleBased_Recorder"
        self.log_path = "supervised_data.jsonl"
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.log_file = open(self.log_path, "a")
        print(f"[{self.name}] Initialized with logging to {self.log_path}")

    def reset(self):
        pass

    def update(self, obs, reward=0.0, done=False, info=None):
        agent_pos = np.array(obs["agent_position"])  # [x, y, z]
        target_pos = np.array(obs["target_position"])  # [x, y, z]
        velocity = np.array(obs["agent_velocity"])  # [x, z]
        forward = np.array(obs["agent_forward_direction"])  # [x, z]
        hp_norm = float(obs["agent_health_normalized"])  # scalar

        # === heading vector (xz) ===
        heading = target_pos[[0, 2]] - agent_pos[[0, 2]]
        heading_norm = np.linalg.norm(heading)
        heading_unit = heading / (heading_norm + 1e-6)

        # === cosine similarity ===
        forward_norm = np.linalg.norm(forward)
        cos_sim = 0.0
        if heading_norm > 1e-6 and forward_norm > 1e-6:
            cos_sim = np.dot(forward, heading) / (heading_norm * forward_norm)

        # === normalized distance to target (max = 30) ===
        distance = np.linalg.norm(target_pos - agent_pos) / 30.0

        # === 最近障礙物方向（nearby_map_objects + terrain_grid） ===
        nearest_obs_dir = None
        min_obs_dist = float("inf")

        # 1. nearby_map_objects
        for obj in obs["nearby_map_objects"]:
            rel = np.array(obj["relative_position"])
            d = np.linalg.norm(rel)
            if d < min_obs_dist and d > 1e-6:
                min_obs_dist = d
                nearest_obs_dir = rel / d

        # 2. terrain_grid
        grid = obs["terrain_grid"]
        for row in grid:
            for cell in row:
                ttype = int(cell["terrain_type"])
                if ttype != 0:
                    rel = np.array(cell["relative_position"])
                    d = np.linalg.norm(rel)
                    if d < min_obs_dist and d > 1e-6:
                        min_obs_dist = d
                        nearest_obs_dir = rel / d

        # fallback
        if nearest_obs_dir is None:
            nearest_obs_dir = np.array([0.0, 0.0])

        # === obs vector ===
        obs_vec = np.array([
            *heading_unit.tolist(),         # (2,)
            distance,                       # (1,)
            *velocity,                      # (2,)
            *forward,                       # (2,)
            cos_sim,                        # (1,)
            hp_norm,                        # (1,)
            *nearest_obs_dir.tolist(),      # (2,)
        ], dtype=np.float32)

        # === 決策動作 ===
        move_vec = heading_unit  # 固定往 target 前進

        self.log_data(obs_vec.tolist(), move_vec.tolist())

        return (
            np.array(move_vec, dtype=np.float32),
            np.array([0, 0], dtype=np.int32)  # 不用道具
        )

    def log_data(self, obs_vec, action_vec):
        self.log_file.write(json.dumps({"obs": obs_vec, "action": action_vec}) + "\n")
