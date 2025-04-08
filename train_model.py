import numpy as np
import random
from collections import deque

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))

    def forward(self, x):
        # Forward pass
        self.input = x
        self.hidden = np.maximum(0, np.dot(x, self.weights_input_hidden) + self.bias_hidden)  # ReLU activation
        self.output = np.dot(self.hidden, self.weights_hidden_output) + self.bias_output
        return self.output

    def backward(self, x, y, target):
        # Compute loss gradient
        output_error = y - target
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * (self.hidden > 0)

        # Update weights and biases
        self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden.T, output_error)
        self.bias_output -= self.learning_rate * np.sum(output_error, axis=0, keepdims=True)
        self.weights_input_hidden -= self.learning_rate * np.dot(x.T, hidden_error)
        self.bias_hidden -= self.learning_rate * np.sum(hidden_error, axis=0, keepdims=True)

    def train(self, x, target):
        y = self.forward(x)
        self.backward(x, y, target)

    def predict(self, x):
        return self.forward(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Replace TensorFlow model with SimpleNN
        self.model = SimpleNN(state_size, 24, action_size, learning_rate=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # Exploit

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                next_q_values = self.model.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(next_q_values[0])

            # Train the model
            self.model.train(state, target)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        """
        Save the model weights to a file.
        """
        np.savez(filename, 
                weights_input_hidden=self.model.weights_input_hidden,
                bias_hidden=self.model.bias_hidden,
                weights_hidden_output=self.model.weights_hidden_output,
                bias_output=self.model.bias_output)
        
    def load(self, filename):
        """
        Load the model weights from a file.
        """
        data = np.load(filename)
        self.model.weights_input_hidden = data['weights_input_hidden']
        self.model.bias_hidden = data['bias_hidden']
        self.model.weights_hidden_output = data['weights_hidden_output']
        self.model.bias_output = data['bias_output']