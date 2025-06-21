import os, sys, math, pickle, random, datetime

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
MAGNIFY = 300.0
WALL_PENALTY = -5.0

class MLPlay:
    def __init__(self, ai_name: str, *args, **kwargs):
        self.player = ai_name
        self.q_table = self.load_q_table()
        self.last_state = None
        self.last_action = None
        self.last_score = 0
        self.last_weights = None

        self.run_counter_path = os.path.join(os.path.dirname(__file__), f"model_r/run_count_{self.player}.txt")
        self.run_count = int(open(self.run_counter_path).read()) if os.path.exists(self.run_counter_path) else 0
        self.start_time = datetime.datetime.now()
        print(f"[{self.player}] RL agent initialized.")
        print(f"[{self.player}] ▶️ Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[{self.player}] Q-table size: {len(self.q_table)} / Run count: {self.run_count}")

        self.reward_log_path = f"ml/model_r/reward_log_{self.player}.csv"
        if not os.path.exists(self.reward_log_path) or os.path.getsize(self.reward_log_path) == 0:
            with open(self.reward_log_path, "w") as f:
                f.write("episode,frame,reward\n")
        self.frame_count = 0

    def load_q_table(self):
        path = os.path.join(os.path.dirname(__file__), f"model_r/q_table_{self.player}.pickle")
        return pickle.load(open(path, "rb")) if os.path.exists(path) else {}

    def get_model_path(self):
        return os.path.join(os.path.dirname(__file__), f"model_r/q_table_{self.player}.pickle")

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

        # 偏移後地圖牆邊懲罰（真實偏移 40,120）
        if sx <= 90: W["LEFT"] += WALL_PENALTY
        if sx >= 1190: W["RIGHT"] += WALL_PENALTY
        if sy <= 170: W["UP"] += WALL_PENALTY
        if sy >= 720: W["DOWN"] += WALL_PENALTY

        self.current_weights = W.copy()
        return tuple(self.quantize(W[d], base=1.0) for d in ACTIONS)

    def choose_action(self, state, epsilon=0.3):
        if random.random() < epsilon:
            return random.choice(ACTIONS)
        q_vals = [self.q_table.get((state, a), 0.0) for a in ACTIONS]
        max_q = max(q_vals)
        best = [a for a, q in zip(ACTIONS, q_vals) if q == max_q]
        return random.choice(best)

    def compute_reward(self, scene_info, last_action):
        reward = 0.0
                
        score_delta = scene_info["score"] - self.last_score
        if score_delta > 0:
            reward += 10.0  # 吃到加分食物
        elif score_delta < 0:
            reward -= 10.0  # 吃到扣分物

        if self.last_weights and self.current_weights:
            if self.current_weights[last_action] > self.last_weights[last_action]:
                reward += 1.0
            elif self.current_weights[last_action] < self.last_weights[last_action]:
                reward -= 1.0

            max_dir = max(self.current_weights, key=self.current_weights.get)
            if self.current_weights[max_dir] > 0:
                if last_action == max_dir:
                    reward += 2.0
                else:
                    reward -= 1.0

        if last_action == self.last_action:
            reward -= 0.5

        return round(max(-10, min(10, reward)), 2)

    def update(self, scene_info: dict, *args, **kwargs):
        if isinstance(scene_info, list): scene_info = scene_info[0]
        if scene_info["status"] != "GAME_ALIVE": return []

        state = self.get_features(scene_info)
        epsilon = max(0.05, 0.3 * (0.995 ** self.run_count))
        action = self.choose_action(state, epsilon)

        if self.last_state and self.last_action is not None:
            reward = self.compute_reward(scene_info, self.last_action)
            alpha, gamma = 0.1, 0.9
            old_q = self.q_table.get((self.last_state, self.last_action), 0.0)
            next_max_q = max([self.q_table.get((state, a), 0.0) for a in ACTIONS])
            self.q_table[(self.last_state, self.last_action)] = old_q + alpha * (reward + gamma * next_max_q - old_q)

            self.frame_count += 1
            with open(self.reward_log_path, "a") as f:
                f.write(f"{self.run_count},{self.frame_count},{reward}\n")

        self.last_state = state
        self.last_action = action
        self.last_score = scene_info["score"]
        self.last_weights = self.current_weights.copy()
        return [action]

    def reset(self):
        end_time = datetime.datetime.now()
        print(f"[{self.player}] Reset and save Q-table.")
        print(f"[{self.player}] ⏱ Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[{self.player}] ⏱ End time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        os.makedirs(os.path.dirname(self.get_model_path()), exist_ok=True)
        with open(self.get_model_path(), "wb") as f:
            pickle.dump(self.q_table, f)

        self.run_count += 1
        with open(self.run_counter_path, "w") as f:
            f.write(str(self.run_count))

        if self.run_count >= 3000:
            print(f"[{self.player}] ✅ Training finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"[{self.player}] 🔚 Reached max training runs, exiting...")
            sys.exit(0)

        self.last_state = None
        self.last_action = None
        self.last_score = 0
        self.last_weights = None
        self.frame_count = 0
