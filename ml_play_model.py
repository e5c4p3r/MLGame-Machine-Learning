import pickle
import numpy as np
from sklearn.metrics import accuracy_score

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.game_width = 200
        self.platform_y = 400
        self.rule_actions_path = "rule_actions.pkl"
        self.model_actions_path = "model_actions.pkl"
        self.rule_actions = []
        self.model_actions = []
        with open("ml/models/knn.pkl", "rb") as f:
            self.model = pickle.load(f)
        self.reset()

    def save_rule_actions(self):
        with open(self.rule_actions_path, "wb") as f:
            pickle.dump(self.rule_actions, f)

    def save_model_actions(self):
        with open(self.model_actions_path, "wb") as f:
            pickle.dump(self.model_actions, f)

    def update(self, scene_info, *args, **kwargs):
        if scene_info["status"] in ["GAME_OVER", "GAME_PASS"]:
            print(f"{scene_info['status']}", flush=True)
            self.save_rule_actions()
            self.save_model_actions()
            return "RESET"

        # Extract state
        ball_x, ball_y = scene_info["ball"]
        platform_x = scene_info["platform"][0]
        frame = scene_info["frame"]

        # Calculate ball speed
        if self.last_ball_position is not None:
            ball_dx = ball_x - self.last_ball_position[0]
            ball_dy = ball_y - self.last_ball_position[1]
        else:
            ball_dx, ball_dy = 0, 0  # Default speed for the first frame

        state = (ball_x, ball_y, ball_dx, ball_dy, platform_x)

        # Update the last ball position
        self.last_ball_position = (ball_x, ball_y)

        # Predict the ball's future position by rules
        if frame < 150:
            target_ball_x = self.initial_ball_x
        else:
            if ball_dy > 0:
                target_ball_x = self.predict_ball_x(ball_x, ball_y, ball_dx, ball_dy)
            else:
                target_ball_x = self.game_width / 2

        # Predict the action using the rules
        if target_ball_x < platform_x:
            rule_dir = -1
        elif target_ball_x > platform_x + 20:  # Platform width is 40
            rule_dir = 1
        else:
            rule_dir = 0

        # Predict the action using the model
        model_dir = self.model.predict(np.array([state]))

        self.rule_actions.append(rule_dir)
        self.model_actions.append(model_dir)

        # Convert the model's prediction to an action
        if model_dir == -1:
            action = "MOVE_LEFT"
        elif model_dir == 1:
            action = "MOVE_RIGHT"
        else:
            action = "NONE"

        return action

    def reset(self):
        self.last_ball_position = None  # Reset the ball position tracker
        self.initial_ball_x = self.game_width / 2

    def predict_ball_x(self, ball_x, ball_y, ball_dx, ball_dy):
        """
        Predict the ball's future position by simulating its movement.
        """

        dist_y = self.platform_y - ball_y
        if ball_dy < 0:
            dist_y += ball_y * 2
        predicted_ball_x = ball_x + (self.platform_y - ball_y) * ((ball_dx / ball_dy) if ball_dy != 0 else ball_x)

        if predicted_ball_x > self.game_width:
            collide_count = predicted_ball_x // self.game_width
            if collide_count % 2 == 0:
                predicted_ball_x = predicted_ball_x % self.game_width
            else:
                predicted_ball_x = self.game_width - (predicted_ball_x % self.game_width)
        elif predicted_ball_x < 0:
            collide_count = (-predicted_ball_x // self.game_width) + 1
            if collide_count % 2 == 0:
                predicted_ball_x = self.game_width + predicted_ball_x % self.game_width
            else:
                predicted_ball_x = -predicted_ball_x % self.game_width

        return predicted_ball_x