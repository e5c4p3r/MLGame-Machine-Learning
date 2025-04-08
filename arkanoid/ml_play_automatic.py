import pickle
import random

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.game_width = 200
        self.platform_y = 400  # Y-coordinate of the platform
        self.data_path = "train_data.pkl"
        self.train_data = []
        self.game_passed = False
        self.reset()
    
    def save_train_data(self):
        # wb: write binary mode
        with open(self.data_path, "wb") as f:
            pickle.dump(self.train_data, f)

    def update(self, scene_info, *args, **kwargs):
        if self.game_passed:
            print(len(self.train_data))
            return "NONE"

        if scene_info["status"] in ["GAME_OVER", "GAME_PASS"]:
            if len(scene_info["bricks"]) == 0 and len(scene_info["hard_bricks"]) == 0:
                self.save_train_data() # Save training data when win
                self.game_passed = True
                print("Game passed!")
            else:
                self.train_data = []
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

        # Predict the ball's future position
        if frame < 150:
            target_ball_x = self.initial_ball_x
        else:
            if ball_dy > 0:
                target_ball_x = self.predict_ball_x(ball_x, ball_y, ball_dx, ball_dy)
            else:
                target_ball_x = self.game_width / 2

        # Decide action based on predicted ball position
        if target_ball_x < platform_x:
            dir = -1
            action = "MOVE_LEFT"
        elif target_ball_x > platform_x + 20:  # Platform width is 40
            dir = 1
            action = "MOVE_RIGHT"
        else:
            dir = 0
            action = "NONE"

        self.train_data.append((state, dir))

        return action

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

    def reset(self):
        print("Resetting...")
        self.last_ball_position = None  # Reset the ball position tracker
        # self.train_data = []
        self.initial_ball_x = random.randrange(self.game_width * 0.2, self.game_width * 0.8)  # Random initial ball position