import pickle
import os

MODEL_PATH = "model/knn_v10.pickle"

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.side = ai_name
        self.ball_served = False
        self.data = []

        self.save_dir = "pingpong_data"
        os.makedirs(self.save_dir, exist_ok=True)
        print(self.side + ": " + MODEL_PATH)

        # Load trained model
        with open(MODEL_PATH, "rb") as f:
            self.knn = pickle.load(f)

    def update(self, scene_info, *args, **kwargs):
        # If the game ends, record the last frame and reset
        if scene_info["status"] != "GAME_ALIVE":
            self.data.append({
                "frame": scene_info["frame"],
                "ball_x": scene_info["ball"][0],
                "ball_y": scene_info["ball"][1],
                "ball_dx": scene_info["ball_speed"][0],
                "ball_dy": scene_info["ball_speed"][1],
                "platform_x": scene_info["platform_1P"][0] if self.side == "1P" else scene_info["platform_2P"][0],
                "command": "NONE",
                "predicted_x": 100,
                "blocker_x": scene_info.get("blocker", [0, 0])[0],
                "blocker_y": scene_info.get("blocker", [0, 0])[1],
                "status": scene_info["status"],
                "side": self.side,
            })
            return "RESET"

        # Extract blocker info
        blocker = scene_info.get("blocker")
        blocker_x = blocker[0] if blocker else 0
        blocker_y = blocker[1] if blocker else 0

        # Serve ball with blocker position awareness
        if not self.ball_served:
            self.ball_served = True
            if blocker_x > 90:
                return "SERVE_TO_LEFT"
            else:
                return "SERVE_TO_RIGHT"

        # === Game information ===
        ball_x, ball_y = scene_info["ball"]
        ball_dx, ball_dy = scene_info["ball_speed"]
        platform_x = scene_info["platform_1P"][0] if self.side == "1P" else scene_info["platform_2P"][0]

        # Normalize to 1P view: ball_y, ball_dy, blocker_y
        view_ball_y = 500 - ball_y if self.side == "2P" else ball_y
        view_ball_dy = -ball_dy if self.side == "2P" else ball_dy
        view_blocker_y = 500 - blocker_y if self.side == "2P" else blocker_y

        # landing_y is always 420 from 1P view
        landing_y = 420

        # === Simulate landing point ===
        predicted_x = self.simulate_landing(
            (ball_x, view_ball_y), (ball_dx, view_ball_dy),
            (blocker_x, view_blocker_y), landing_y
        )

        # === Prepare features ===
        feature = [
            ball_x,
            view_ball_y,
            ball_dx,
            view_ball_dy,
            platform_x,
            predicted_x,
            blocker_x,
            view_blocker_y
        ]

        # Predict command from model
        command = self.knn.predict([feature])[0]

        # === Save data for training ===
        self.data.append({
            "frame": scene_info["frame"],
            "ball_x": ball_x,
            "ball_y": ball_y,
            "ball_dx": ball_dx,
            "ball_dy": ball_dy,
            "platform_x": platform_x,
            "command": command,
            "predicted_x": predicted_x,
            "blocker_x": blocker_x,
            "blocker_y": blocker_y,
            "status": scene_info["status"],
            "side": self.side
        })

        return command

    def simulate_landing(self, ball_pos, ball_speed, blocker, landing_y):
        x, y = ball_pos
        dx, dy = ball_speed

        if blocker:
            bx, by = blocker
            bw, bh, bd = 30, 20, 5  # blocker width/height/speed
        else:
            bx = by = bw = bh = bd = 0

        while True:
            x += dx
            y += dy

            # Bounce off walls
            if x <= 0 or x >= 200:
                dx *= -1
                x = max(0, min(x, 200))

            # Bounce off top/bottom
            if y <= 0 or y >= 420:
                dy *= -1
                y = max(0, min(y, 420))

            # Check blocker collision
            if blocker and (bx <= x <= bx + bw) and (by <= y <= by + bh):
                dx *= -1

            # Move blocker
            if blocker:
                bx += bd
                if bx <= 0 or bx + bw >= 200:
                    bd *= -1
                    bx = max(0, min(bx, 200 - bw))

            # Stop when reaching landing_y
            if (dy > 0 and y >= landing_y) or (dy < 0 and y <= landing_y):
                break

        return x

    def reset(self):
        if self.data:
            filename = os.path.join(self.save_dir, f"{self.side}_data.pickle")
            with open(filename, "ab") as f:
                pickle.dump(self.data, f)
            print(f"[INFO] Saved {len(self.data)} frames to {filename}")
            self.data = []

        self.ball_served = False
