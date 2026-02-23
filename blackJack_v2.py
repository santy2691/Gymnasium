import os
import random
import gymnasium as gym
import torch
from torch import nn

class Agente(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

def preprocess_state(state):
    return torch.tensor([state[0], state[1], float(state[2])], dtype=torch.float32)

def main():
    # Hyperparams
    learning_rate = 0.01
    n_episodes = 10000
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("Blackjack-v1", sab=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = Agente().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    os.makedirs("weights", exist_ok=True)

    for episode in range(n_episodes):
        obs, info = env.reset()
        state = preprocess_state(obs).to(device)
        done = False
        total_reward = 0

        while not done:
            q_values = agent(state)  # shape (2,)
            if random.random() > epsilon:
                action = torch.argmax(q_values).item()
            else:
                action = env.action_space.sample()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if not done:
                with torch.no_grad():
                    next_state = preprocess_state(next_obs).to(device)
                    next_q_values = agent(next_state)
                    max_next_q = torch.max(next_q_values)
            else:
                max_next_q = torch.tensor(0.0, device=device)

            target_value = reward + gamma * max_next_q.item() # type: ignore
            current_q = q_values[action]

            target_tensor = torch.tensor(target_value, dtype=current_q.dtype, device=current_q.device)
            loss = loss_fn(current_q, target_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reward += reward # type: ignore
            state = preprocess_state(next_obs).to(device)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}: reward={total_reward}, epsilon={epsilon:.3f}")

    env.close()
    torch.save(agent.state_dict(), 'weights/modelo_agente_v2.pth')
    print("Modelo guardado en weights/modelo_agente_v2.pth")

if __name__ == "__main__":
    main()