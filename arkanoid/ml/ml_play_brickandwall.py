"""
brick and wall
"""
import pickle
import os
import random
import sys

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.ball_served = False
        self.previous_ball_position = (0, 0)
        self.game_data = []
        self.previous_velocity = (0, 0)  
        self.current_level = self.get_level_from_args()  
        self.waiting_zone = 100  
        self.low_bricks = []  
        self.serve_position = None  # **Initialize serve position**
        self.serve_direction = None  # **Initialize serve direction**
        self.previous_command = "NONE"  # Initialize previous frame's command

        self.run_count = 0  # **Track total execution count**
        self.successful_data_count = 0  # **Track valid data count (successful completion or 1 remaining brick)**
        self.max_runs = 100  # **Maximum execution count**
        self.max_successful_data = 50  # **Stop when this many successful data points are collected**

    def get_level_from_args(self):
        """Retrieve --level argument from the command line"""
        for i in range(len(sys.argv)):
            if sys.argv[i] == "--level" and i + 1 < len(sys.argv):
                return int(sys.argv[i + 1])  
        return 0  

    def update(self, scene_info, *args, **kwargs):
        """ **Enhance smooth movement + adjust landing prediction** """
        scene_info["level"] = self.current_level

        if scene_info["status"] in ("GAME_OVER", "GAME_PASS"):
            return "RESET"

        ball_x, ball_y = scene_info["ball"]
        platform_x = scene_info["platform"][0]
        bricks = scene_info["bricks"]

        # **Serve (random left/middle/right + serve direction), ensure platform moves to serve position first**
        if not self.ball_served:
            if self.serve_position is None:
                serve_positions = [30, 65, 100, 135, 170]  # Left, slightly left, center, slightly right, right
                self.serve_position = random.choice(serve_positions)
                # print(self.serve_position)

            # **Move platform to serve position**
            if platform_x + 20 < self.serve_position:
                return "MOVE_RIGHT"
            elif platform_x + 20 > self.serve_position:
                return "MOVE_LEFT"

            # **Once in position, randomly choose serve direction**
            self.serve_direction = random.choice(["SERVE_TO_LEFT", "SERVE_TO_RIGHT"])

            self.ball_served = True  # **Ensure ball is served only once**
            print(f"Serve Position: {self.serve_position}, Serve Direction: {self.serve_direction}")

            return self.serve_direction

        # **Calculate ball velocity**
        prev_ball_x, prev_ball_y = self.previous_ball_position
        delta_x = ball_x - prev_ball_x
        delta_y = ball_y - prev_ball_y

        velocity_changed = self.previous_velocity != (delta_x, delta_y)
        self.previous_velocity = (delta_x, delta_y)

        # **Update low brick list**
        self.low_bricks = [b for b in bricks if b[1] > 300]  # Only consider bricks below 300

        # **Predict landing position**
        predicted_x = self.predict_landing_x(ball_x, ball_y, delta_x, delta_y)
        if velocity_changed:
            predicted_x = self.predict_landing_x(ball_x, ball_y, delta_x, delta_y)

        # **Check if the ball will hit low bricks**
        predicted_x = self.check_low_brick_collision(ball_x, ball_y, delta_x, delta_y, predicted_x)

        # **Dynamically adjust waiting zone**
        self.waiting_zone = 120 if abs(delta_x) > 5 else 80  

        # **Platform waiting strategy (smoother movement)**
        if ball_y > 350:  
            final_target = predicted_x  # **Ball is about to land, fully align with landing position**
        elif ball_y > 250:  
            final_target = (predicted_x + platform_x) // 2  # **Gradually reduce error**
        else:  
            final_target = self.waiting_zone  

        # **Platform movement decision**
        move_threshold = max(2, min(5, 10 - abs(delta_y) * 2))  
        if platform_x + 20 < final_target - move_threshold:
            command = "MOVE_RIGHT"
        elif platform_x + 20 > final_target + move_threshold:
            command = "MOVE_LEFT"
        else:
            command = "NONE"

        # **Calculate actual landing position**
        actual_x = ball_x + ((400 - ball_y) // abs(delta_y)) * delta_x if delta_y != 0 else ball_x

        # **Check if the ball hit a brick**
        brick_impact = any(abs(ball_x - bx) < 10 and abs(ball_y - by) < 10 for bx, by in scene_info["bricks"])

        # **Check if the ball hit the wall**
        wall_impact = ball_x <= 0 or ball_x >= 200

        # **Calculate ball landing time**
        time_to_land = (400 - ball_y) // abs(delta_y) if delta_y != 0 else 0

        # **Store data**
        frame_data = {
            "level": self.current_level,
            "frame": scene_info["frame"],
            "ball_x": ball_x,
            "ball_y": ball_y,
            "delta_x": delta_x,
            "delta_y": delta_y,
            "platform_x": platform_x,
            "prev_platform_x": self.previous_ball_position[0],  # **Previous frame's platform position**
            "predicted_x": predicted_x,
            "actual_x": actual_x,  # **Actual landing position**
            "time_to_land": time_to_land,  # **Time remaining for ball to land**
            "brick_impact": int(brick_impact),  # **Did it hit a brick?**
            "wall_impact": int(wall_impact),  # **Did it hit a wall?**
            "bounce_count": scene_info.get("ball_bounce", 0),  # **Ball bounce count**
            "command": command,
            "prev_command": self.previous_command,  # **Previous frame's command**
            "serve_position": self.serve_position,
            "serve_direction": self.serve_direction,
            "game_state": scene_info["status"],
            "remain_brick": len(scene_info["bricks"]),
        }

        # **Update previous frame information**
        self.previous_command = command
        self.previous_ball_position = (ball_x, ball_y)

        # **Save data**
        self.game_data.append(frame_data)

        return command

    def predict_landing_x(self, ball_x, ball_y, delta_x, delta_y):
        """ **Predict the ball's landing position, considering wall bounces** """
        platform_y = 400  

        if delta_y == 0:
            return ball_x  
    
        steps_to_ground = (platform_y - ball_y) // abs(delta_y)
        predicted_x = ball_x + steps_to_ground * delta_x  

        # **Handle wall bounces**
        while predicted_x < 0 or predicted_x > 200:
            if predicted_x < 0:
                predicted_x = -predicted_x  
            elif predicted_x > 200:
                predicted_x = 400 - predicted_x  

        return predicted_x

    def check_low_brick_collision(self, ball_x, ball_y, delta_x, delta_y, predicted_x):
        """ **If the ball might hit low bricks, adjust landing position** """
        for brick_x, brick_y in self.low_bricks:
            if abs(predicted_x - brick_x) < 25 and 300 < brick_y < 380:  
                # **Simulate brick bounce effect**
                if delta_y > 0:  # Ball moving downward
                    predicted_x -= delta_x * 2  
                else:  # Ball moving upward
                    predicted_x += delta_x * 2  
        return predicted_x  

    def reset(self):
        """ **Save data and proceed to the next level** """
        self.ball_served = False
        self.serve_position = None
        self.serve_direction = None

        self.run_count += 1
        
        if self.game_data:
            last_frame = self.game_data[-1]

            # **Save only successful data (GAME_PASS) or when 1 brick remains**
            if last_frame["game_state"] == "GAME_PASS" or last_frame["remain_brick"] == 1:
                file_path = f"pickles_10050_wallandbrick/game_data_levels_{self.current_level}.pickle"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "ab") as f:  # **Use 'ab' mode (append mode)**
                    pickle.dump(self.game_data, f)

                self.successful_data_count += 1  # **Count successful data**
                # print(f"✅ Saved successful data for level {self.current_level} ({self.successful_data_count}/30)")

        self.game_data = []

        # **Terminate when reaching max runs or collecting enough successful data**
        if self.run_count >= self.max_runs or self.successful_data_count >= self.max_successful_data:
            print("📌 Data collection complete, reached the limit, terminating program.")
            exit()
