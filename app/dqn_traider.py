import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from keras.models import load_model
import matplotlib.pyplot as plt

# === 1. Charger le modèle LSTM ===
lstm_model = load_model("lstm_model.h5")

# Déterminer dynamiquement la taille de la fenêtre attendue par le LSTM
expected_window_size = lstm_model.input_shape[1]

# === 2. Environnement avec LSTM intégré ===
class MarketEnv:
    def __init__(self, prices, window_size=expected_window_size, lstm_model=None):
        self.prices = prices
        self.window_size = window_size
        self.lstm_model = lstm_model
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.total_profit = 0
        self.position = 0
        self.entry_price = 0
        return self._get_state()

    def _get_state(self):
        window = self.prices[self.current_step - self.window_size:self.current_step]
        window_array = np.array(window).reshape((1, self.window_size, 1))

        if self.lstm_model:
            prediction = self.lstm_model.predict(window_array, verbose=0)[0][0]
        else:
            prediction = 0.0

        state = np.append(window, prediction)
        return state

    def step(self, action):
        reward = 0
        done = False
        price = self.prices[self.current_step]

        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 1:
            profit = price - self.entry_price
            reward = profit
            self.total_profit += profit
            self.position = 0

        self.current_step += 1
        if self.current_step >= len(self.prices) - 1:
            done = True

        next_state = self._get_state()
        return next_state, reward, done

# === 3. Réseau DQN ===
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# === 4. Agent DQN ===
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_model(next_state)).item()
            target_f = self.model(state)
            target_f[action] = target
            output = self.model(state)
            loss = self.loss_fn(output, target_f.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# === 5. Entraînement du DQN ===
def train_dqn(prices, lstm_model, episodes=50):
    window_size = expected_window_size
    env = MarketEnv(prices, window_size, lstm_model)
    agent = Agent(state_size=window_size + 1, action_size=3)
    scores = []

    for e in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay(32)
            if done:
                agent.update_target_model()
                break

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        scores.append(env.total_profit)
        print(f"Episode {e+1}/{episodes} - Profit: {env.total_profit:.2f} - Epsilon: {agent.epsilon:.4f}")

    # Courbe d'apprentissage
    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Total Profit")
    plt.title("Performance DQN avec LSTM intégré")
    plt.grid()
    plt.show()

# === 6. Lancement ===
if __name__ == "__main__":
    np.random.seed(42)
    prices = np.sin(np.linspace(0, 100, 1000)) * 10 + 100  # Données simulées
    train_dqn(prices, lstm_model)
