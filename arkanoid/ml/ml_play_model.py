import pickle
import numpy as np
import os
import random

MODEL_PATH = "pickles_model/trained_knn_model_wallandbrick.pickle"  # **Load the trained KNN model**

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.model = self.load_model()
        self.ball_served = False
        self.previous_ball_position = (0, 0)
        self.previous_velocity = (0, 0)
        self.previous_command = "NONE"
        self.serve_position = None  # **Serving position**
        self.serve_direction = None  # **Serving direction**

    def load_model(self):
        """ **Load the trained KNN model** """
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        else:
            print("⚠️ Unable to load KNN model! Please train the model first!")
            return None

    def update(self, scene_info, *args, **kwargs):
        """ **Use the KNN model for inference** """
        if scene_info["status"] in ("GAME_OVER", "GAME_PASS"):
            self.ball_served = False
            self.serve_position = None
            self.serve_direction = None
            return "RESET"

        ball_x, ball_y = scene_info["ball"]
        platform_x = scene_info["platform"][0]

        # **Random serving (ensure the platform moves to the designated serve position)**
        if not self.ball_served:
            if self.serve_position is None:
                serve_positions = [30, 100, 170]  # **Left, Middle, Right**
                self.serve_position = random.choice(serve_positions)

            # **Move the platform to the serve position**
            if platform_x + 20 < self.serve_position:
                return "MOVE_RIGHT"
            elif platform_x + 20 > self.serve_position:
                return "MOVE_LEFT"

            # **Select the serve direction**
            self.serve_direction = random.choice(["SERVE_TO_LEFT", "SERVE_TO_RIGHT"])

            self.ball_served = True  # **Serve the ball**
            print(f"Serve Position: {self.serve_position}, Serve Direction: {self.serve_direction}")

            return self.serve_direction

        # **Calculate the ball's movement direction**
        prev_ball_x, prev_ball_y = self.previous_ball_position
        delta_x = ball_x - prev_ball_x
        delta_y = ball_y - prev_ball_y

        # **Predict the landing position**
        predicted_x = self.predict_landing_x(ball_x, ball_y, delta_x, delta_y)

        # **Calculate the remaining time until the ball lands**
        time_to_land = (400 - ball_y) // abs(delta_y) if delta_y != 0 else 0

        # **Construct the feature vector**
        feature_vector = np.array([[
            ball_x, ball_y, delta_x, delta_y,
            platform_x, predicted_x, time_to_land
        ]])

        # **Use the KNN model for prediction**
        if self.model:
            predicted_command = self.model.predict(feature_vector)[0]
        else:
            predicted_command = "NONE"

        self.previous_ball_position = (ball_x, ball_y)
        return predicted_command

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
        """ **If the ball is likely to collide with low bricks, adjust the predicted landing position in advance** """
        for brick_x, brick_y in self.low_bricks:
            if abs(predicted_x - brick_x) < 25 and 300 < brick_y < 380:  
                # **Simulate the effect of bouncing off bricks**
                if delta_y > 0:  # Ball moving downward
                    predicted_x -= delta_x * 2  
                else:  # Ball moving upward
                    predicted_x += delta_x * 2  
        return predicted_x  

    def reset(self):
        """ **Reset the game state** """
        self.ball_served = False
        self.serve_position = None
        self.serve_direction = None
