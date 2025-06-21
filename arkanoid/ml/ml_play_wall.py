"""
wall only 
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
        self.safe_zone = 100  # Default platform waiting safe zone
        self.serve_position = None  # **Initialize serve position**
        self.serve_direction = None  # **Initialize serve direction**
        self.previous_command = "NONE"  # Initialize previous command

    def get_level_from_args(self):
        """Retrieve --level argument from the command line"""
        for i in range(len(sys.argv)):
            if sys.argv[i] == "--level" and i + 1 < len(sys.argv):
                return int(sys.argv[i + 1])  
        return 0  
    
    def update(self, scene_info, *args, **kwargs):
        """ Smooth platform movement for better ball catching """
        scene_info["level"] = self.current_level

        if scene_info["status"] in ("GAME_OVER", "GAME_PASS"):
            return "RESET"

        ball_x, ball_y = scene_info["ball"]
        platform_x = scene_info["platform"][0]
        
        # **Serve (random left/middle/right + serve direction), ensure platform moves to serve position first**
        if not self.ball_served:
            if self.serve_position is None:
                serve_positions = [30, 100, 170]  # Left, Middle, Right
                self.serve_position = random.choice(serve_positions)
                print(self.serve_position)

            # **Move platform to serve position**
            if platform_x + 20 < self.serve_position:
                return "MOVE_RIGHT"
            elif platform_x + 20 > self.serve_position:
                return "MOVE_LEFT"

            # **Once in position, randomly choose serve direction**
            if self.serve_position == 30:  # Left serve position
                self.serve_direction = random.choices(["SERVE_TO_RIGHT", "SERVE_TO_LEFT"], weights=[0.8, 0.2])[0]
            elif self.serve_position == 170:  # Right serve position
                self.serve_direction = random.choices(["SERVE_TO_LEFT", "SERVE_TO_RIGHT"], weights=[0.8, 0.2])[0]
            else:  # Middle serve position (completely random)
                self.serve_direction = random.choice(["SERVE_TO_LEFT", "SERVE_TO_RIGHT"])

            self.ball_served = True  # **Ensure ball is served only once**
            return self.serve_direction

        # Calculate ball movement direction
        prev_ball_x, prev_ball_y = self.previous_ball_position
        delta_x = ball_x - prev_ball_x
        delta_y = ball_y - prev_ball_y

        # **Check ball velocity change**
        velocity_changed = self.previous_velocity != (delta_x, delta_y)
        self.previous_velocity = (delta_x, delta_y)

        # **Predict landing position**
        predicted_x = self.predict_landing_x(ball_x, ball_y, delta_x, delta_y)

        # **If velocity changes, recalculate landing position**
        if velocity_changed:
            predicted_x = self.predict_landing_x(ball_x, ball_y, delta_x, delta_y)

        # **Platform waiting strategy**
        if ball_y > 350:  # Ball is about to land, align platform with landing position
            final_target = predicted_x
        elif ball_y > 250:  # Ball is descending, gradually move platform towards landing position
            final_target = (predicted_x + platform_x) // 2
        else:  # Ball is still high, platform waits in a safe zone away from walls
            final_target = self.safe_zone

        # **Platform movement decision**
        move_threshold = max(3, min(6, 10 - abs(delta_y) * 2))  
        if platform_x + 20 < final_target - move_threshold:
            command = "MOVE_RIGHT"
        elif platform_x + 20 > final_target + move_threshold:
            command = "MOVE_LEFT"
        else:
            command = "NONE"

        # **Store data**
        # **Calculate actual landing position**
        actual_x = ball_x + ((400 - ball_y) // abs(delta_y)) * delta_x if delta_y != 0 else ball_x

        # **Check if the ball hits a brick**
        brick_impact = any(abs(ball_x - bx) < 10 and abs(ball_y - by) < 10 for bx, by in scene_info["bricks"])

        # **Check if the ball hits the wall**
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
        """
        Predict the ball's landing position, considering wall bounces
        """
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

    def reset(self):
        """ Save data and move to the next level """
        self.ball_served = False
        self.serve_position = None
        self.serve_direction = None
        
        if self.game_data and self.game_data[-1]["level"] == self.current_level:
            file_path = f"pickles_stable_wall/game_data_levels_{self.current_level}.pickle"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                pickle.dump(self.game_data, f)
            print(f"✅ Saved game data for level {self.current_level}")

            self.game_data = []
