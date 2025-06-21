import os
import pickle
import math

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
MAGNIFY = 300.0  # 與訓練時一致的放大倍率

class MLPlay:
    def __init__(self, ai_name: str, *args, **kwargs):
        self.player = ai_name
        self.q_table = self.load_q_table()
        print(f"[{self.player}] Model play initialized. Q-table size: {len(self.q_table)}")

    def load_q_table(self):
        model_path = os.path.join(os.path.dirname(__file__), f"modelss/q_table_{self.player}.pickle")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                return pickle.load(f)
        print(f"[{self.player}] ❌ No trained model found. Using empty table.")
        return {}

    def quantize(self, x, base=1.0):
        return min(10, int(x // base))

    def get_features(self, scene_info):
        if isinstance(scene_info, list): scene_info = scene_info[0]
        sx, sy = scene_info["self_x"], scene_info["self_y"]

        W = {d: 0.0 for d in ACTIONS}
        for obj in scene_info["foods"]:
            ox, oy, s = obj["x"], obj["y"], obj["score"]
            if not (0 <= ox <= 1200 and 0 <= oy <= 650): continue
            dx, dy = ox - sx, oy - sy
            dist = math.hypot(dx, dy) + 1e-5
            angle = math.degrees(math.atan2(-dy, dx)) % 360

            weight = (s / dist) * MAGNIFY

            if 45 <= angle < 135:
                W["UP"] += weight
            elif 135 <= angle < 225:
                W["LEFT"] += weight
            elif 225 <= angle < 315:
                W["DOWN"] += weight
            else:
                W["RIGHT"] += weight

        # 靠牆懲罰方向
        if sy <= 200: W["UP"] = -999999
        if sy >= 450: W["DOWN"] = -999999
        if sx <= 200: W["LEFT"] = -999999
        if sx >= 1000: W["RIGHT"] = -999999

        return tuple(self.quantize(W[d], base=1.0) for d in ACTIONS)

    def update(self, scene_info, *args, **kwargs):
        if isinstance(scene_info, list): scene_info = scene_info[0]
        if scene_info["status"] != "GAME_ALIVE":
            return []

        state = self.get_features(scene_info)
        q_vals = [self.q_table.get((state, a), 0.0) for a in ACTIONS]
        max_q = max(q_vals)
        best_actions = [a for a, q in zip(ACTIONS, q_vals) if q == max_q]
        return [best_actions[0]] if best_actions else ["UP"]  # 預設動作避免空值

    def reset(self):
        pass
