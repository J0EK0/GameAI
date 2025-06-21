import pickle
import os
import random

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.side = ai_name
        self.ball_served = False
        self.previous_command = "NONE"
        self.previous_ball = (0, 0)
        self.data = []

        self.save_dir = "pingpong_data"
        os.makedirs(self.save_dir, exist_ok=True)

    def update(self, scene_info, *args, **kwargs):
        if scene_info["status"] != "GAME_ALIVE":
            # Still record the final game status frame
            self.data.append({
                "frame": scene_info["frame"],
                "ball_x": scene_info["ball"][0],
                "ball_y": scene_info["ball"][1],
                "ball_dx": scene_info["ball_speed"][0],
                "ball_dy": scene_info["ball_speed"][1],
                "platform_x": scene_info["platform_1P"][0] if self.side == "1P" else scene_info["platform_2P"][0],
                "command": "NONE",
                "predicted_x": 100,  # No need to predict landing point
                "blocker_x": scene_info.get("blocker", [0, 0])[0],
                "blocker_y": scene_info.get("blocker", [0, 0])[1],
                "status": scene_info["status"],
                "side": self.side,
            })
            return "RESET"

        if not self.ball_served:
            self.ball_served = True
            return random.choice(["SERVE_TO_LEFT", "SERVE_TO_RIGHT"])

        # === Extract game state info ===
        ball_x, ball_y = scene_info["ball"]
        ball_dx, ball_dy = scene_info["ball_speed"]
        platform_x = scene_info["platform_1P"][0] if self.side == "1P" else scene_info["platform_2P"][0]
        platform_center = platform_x + 20
        landing_y = 420
        blocker = scene_info.get("blocker")
        blocker_x, blocker_y = blocker if blocker else (0, 0)

        # === Convert to 1P perspective if side is 2P ===
        if self.side == "2P":
            ball_y = 500 - ball_y
            ball_dy = -ball_dy
            blocker_y = 500 - blocker_y

        # Determine ball movement direction
        ball_from_me = ball_dy < 0
        ball_to_me = not ball_from_me
        half_y = 250

        # simulate_landing returns predicted_x and whether it hits the blocker
        if ball_to_me:
            predicted_x, will_hit_blocker = self.simulate_landing(
                (ball_x, ball_y), (ball_dx, ball_dy), (blocker_x, blocker_y), landing_y
            )
        elif ball_from_me and ball_y < half_y:
            predicted_x, will_hit_blocker = self.simulate_landing(
                (ball_x, ball_y), (-ball_dx, -ball_dy), (blocker_x, blocker_y), landing_y
            )
        else:
            predicted_x, will_hit_blocker = self.simulate_landing(
                (ball_x, ball_y), (ball_dx, -ball_dy), (blocker_x, blocker_y), landing_y
            )

        # Offset prediction if it will hit the blocker and platform can move enough
        frames_to_impact = abs(landing_y - ball_y) // max(abs(ball_dy), 1)
        distance_to_target = abs(predicted_x - platform_center)
        max_platform_move = frames_to_impact * 5

        if will_hit_blocker and distance_to_target > 15:
            offset = 15 if platform_center < predicted_x else -15
            predicted_x = max(0, min(predicted_x + offset, 200))

        # Control platform based on predicted position
        tolerance = 2
        if distance_to_target > max_platform_move:
            command = "MOVE_RIGHT" if platform_center < predicted_x else "MOVE_LEFT"
        else:
            if platform_center < predicted_x - tolerance:
                command = "MOVE_RIGHT"
            elif platform_center > predicted_x + tolerance:
                command = "MOVE_LEFT"
            else:
                command = "NONE"

        # === Record training data ===
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
            "side": self.side,
        })

        self.previous_command = command
        self.previous_ball = (ball_x, ball_y)
        return command

    def simulate_landing(self, ball_pos, ball_speed, blocker, landing_y):
        x, y = ball_pos
        dx, dy = ball_speed
        will_hit_blocker = False

        if blocker:
            bx, by = blocker
            blocker_width = 30
            blocker_height = 20
            blocker_dx = 5
        else:
            bx = by = blocker_width = blocker_height = blocker_dx = 0

        while True:
            x += dx
            y += dy

            if x <= 0 or x >= 200:
                dx *= -1
                x = max(0, min(x, 200))

            if y <= 0 or y >= 420:
                dy *= -1
                y = max(0, min(y, 420))

            # Collision with blocker?
            if blocker and (bx <= x <= bx + blocker_width) and (by <= y <= by + blocker_height):
                will_hit_blocker = True
                dx *= -1

            if blocker:
                bx += blocker_dx
                if bx <= 0 or bx + blocker_width >= 200:
                    blocker_dx *= -1
                    bx = max(0, min(bx, 200 - blocker_width))

            if (dy > 0 and y >= landing_y) or (dy < 0 and y <= landing_y):
                break

        return x, will_hit_blocker

    def reset(self):
        if self.data:
            filename = os.path.join(self.save_dir, f"{self.side}_data.pickle")
            with open(filename, "ab") as f:
                pickle.dump(self.data, f)
            print(f"[INFO] Saved {len(self.data)} frames to {filename}")
            self.data = []

        self.ball_served = False
        self.previous_command = "NONE"
