import pickle
import os

MODEL_PATH = "ml/model_P1P2_F74111144.pickle"

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.side = ai_name
        self.ball_served = False

        print(self.side + ": " + MODEL_PATH)

        # Load trained model
        with open(MODEL_PATH, "rb") as f:
            self.knn = pickle.load(f)

    def update(self, scene_info, *args, **kwargs):
        # If the game ends, record the last frame and reset
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"

        # Extract blocker info
        blocker = scene_info.get("blocker")
        blocker_x = blocker[0] if blocker else 0
        blocker_y = blocker[1] if blocker else 0

        # Serve ball with blocker position awareness
        if not self.ball_served:
            self.ball_served = True
            if blocker_x < 90:
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
        view_blocker_y = blocker_y

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
        
        self.ball_served = False
