import os
import torch
import numpy as np
from typing import Dict, Any
from ml.ppo_mlplay_template import PPOModel, ObservationProcessor, ActionProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLPlay:
    """
    僅推論用 PPO Agent，禁止訓練與 buffer 收集。
    """
    def __init__(self, action_space_info=None):
        self.name = "PPO_InferenceAgent"
        self.player_id = 1

        self.obs_proc = ObservationProcessor()
        self.act_proc = ActionProcessor()

        self.model = PPOModel(input_dim=self.obs_proc.get_size()).to(device)
        model_path = f"./models_c/player1/model_latest.pt"
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"[PPO] ✅ Loaded model from {model_path}")
            except Exception as e:
                print(f"[PPO] ❌ Failed to load model: {e}")
        else:
            print(f"[PPO] ❌ Model not found at: {model_path}")
            raise FileNotFoundError(f"{model_path} not found.")

        self.model.eval()
        print("[MLPlay] Inference-only agent initialized")

    def reset(self):
        pass  # 不需重設任何狀態

    def update(self, obs: Dict[str, Any], reward=0.0, done=False, info=None):
        with torch.no_grad():
            obs_tensor = torch.tensor(
                self.obs_proc.process(obs), dtype=torch.float32, device=device
            ).unsqueeze(0)

            action_tensor, _, _ = self.model.act_and_eval(obs_tensor)
            raw_action = action_tensor.squeeze(0).cpu().numpy()
            action = self.act_proc.create_action(raw_action)

        return action
