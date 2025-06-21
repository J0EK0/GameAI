import os, pickle, random, math

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
MAGNIFY = 300.0

class MLPlay:
    def __init__(self, ai_name: str, *args, **kwargs):
        self.player = ai_name
        self.q_table = self.load_q_table()
        self.last_state = None
        self.last_weights = None
        print(f"[{self.player}] ▶️ Model-only mode loaded. Q-table size: {len(self.q_table)}")

    def load_q_table(self):
        path = os.path.join(os.path.dirname(__file__), f"model_r/q_table_{self.player}.pickle")
        return pickle.load(open(path, "rb")) if os.path.exists(path) else {}

    def quantize(self, x, base=1.0):
        return min(10, int(x // base))

    def get_features(self, scene_info):
        if isinstance(scene_info, list):
            scene_info = scene_info[0]
        sx, sy = scene_info["self_x"], scene_info["self_y"]

        W = {d: 0.0 for d in ACTIONS}
        for obj in scene_info["foods"]:
            ox, oy, s = obj["x"], obj["y"], obj["score"]
            if not (0 <= ox <= 1200 and 0 <= oy <= 650): continue

            dx, dy = ox - sx, oy - sy
            dist = math.hypot(dx, dy) + 1e-5

            if dist < 100:
                scale = 3.0
            elif dist < 200:
                scale = 2.0
            else:
                scale = 1.0

            weight = (s / dist) * scale * MAGNIFY

            if sx - 75 <= ox <= sx + 75 and sy - 600 <= oy < sy:
                W["UP"] += weight
            elif sx - 75 <= ox <= sx + 75 and sy < oy <= sy + 600:
                W["DOWN"] += weight
            elif sy - 50 <= oy <= sy + 50 and sx - 650 <= ox < sx:
                W["LEFT"] += weight
            elif sy - 50 <= oy <= sy + 50 and sx < ox <= sx + 650:
                W["RIGHT"] += weight

        # 偏移後地圖牆邊懲罰
        if sx <= 90: W["LEFT"] += -5.0
        if sx >= 1190: W["RIGHT"] += -5.0
        if sy <= 170: W["UP"] += -5.0
        if sy >= 720: W["DOWN"] += -5.0

        self.last_weights = W.copy()
        return tuple(self.quantize(W[d], base=1.0) for d in ACTIONS)

    def choose_action(self, state):
        q_vals = [self.q_table.get((state, a), 0.0) for a in ACTIONS]
        max_q = max(q_vals)
        best = [a for a, q in zip(ACTIONS, q_vals) if q == max_q]
        return random.choice(best)

    def update(self, scene_info: dict, *args, **kwargs):
        if isinstance(scene_info, list): scene_info = scene_info[0]
        if scene_info["status"] != "GAME_ALIVE": return []
        state = self.get_features(scene_info)
        self.last_state = state
        return [self.choose_action(state)]

    def reset(self):
        self.last_state = None
        self.last_weights = None
