import os, sys, math, pickle, datetime

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
MAGNIFY = 300.0
WALL_PENALTY = -5.0

class MLPlay:
    def __init__(self, ai_name: str, *args, **kwargs):
        self.player = ai_name
        self.q_table = self.load_q_table()
        self.last_state = None
        self.last_action = None

    def load_q_table(self):
        path = os.path.join(os.path.dirname(__file__), f"model_re1/q_table_{self.player}.pickle")
        return pickle.load(open(path, "rb")) if os.path.exists(path) else {}

    def quantize(self, x, base=1.0):
        return min(10, int(x // base))

    def get_features(self, scene_info):
        if isinstance(scene_info, list): scene_info = scene_info[0]
        sx, sy = scene_info["self_x"], scene_info["self_y"]
        slv = scene_info.get("self_lv", 1)
        opx, opy = scene_info.get("opponent_x", -999), scene_info.get("opponent_y", -999)
        op_lv = scene_info.get("opponent_lv", 1)

        W = {d: 0.0 for d in ACTIONS}
        all_objects = scene_info["foods"][:]

        # 把敵人視為一個物件加入
        enemy_score = 12 if slv > op_lv else -12
        all_objects.append({"x": opx, "y": opy, "score": enemy_score})

        for obj in all_objects:
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

        # 牆邊修正（根據視窗偏移後的場地範圍）
        if sx <= 90: W["LEFT"] += WALL_PENALTY
        if sx >= 1190: W["RIGHT"] += WALL_PENALTY
        if sy <= 170: W["UP"] += WALL_PENALTY
        if sy >= 720: W["DOWN"] += WALL_PENALTY

        return tuple(self.quantize(W[d], base=1.0) for d in ACTIONS)

    def choose_action(self, state):
        q_vals = [self.q_table.get((state, a), 0.0) for a in ACTIONS]
        return ACTIONS[q_vals.index(max(q_vals))]

    def update(self, scene_info: dict, *args, **kwargs):
        if isinstance(scene_info, list): scene_info = scene_info[0]
        if scene_info["status"] != "GAME_ALIVE": return []

        state = self.get_features(scene_info)
        action = self.choose_action(state)
        self.last_state = state
        self.last_action = action
        return [action]

    def reset(self):
        self.last_state = None
        self.last_action = None
