import os
import random
import gymnasium as gym
import torch
from torch import nn
from collections import deque # Para la memoria

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
    # Convertimos el Booleano del As en 1.0 o 0.0
    return torch.tensor([state[0], state[1], float(state[2])], dtype=torch.float32)

def main():
    # Hyperparams
    learning_rate = 0.001 # Bajamos un poco el LR para mayor estabilidad
    n_episodes = 10000
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9995
    batch_size = 64
    memory_size = 10000
    target_update = 10 # Cada cuántos episodios actualizamos la red espejo

    device = "mps"

    env = gym.make("Blackjack-v1", sab=False)
    
    # Redes: La que aprende (policy) y la que da estabilidad (target)
    policy_net = Agente().to(device)
    target_net = Agente().to(device)
    target_net.load_state_dict(policy_net.state_dict()) # Copiamos pesos iniciales
    
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = deque(maxlen=memory_size) # Nuestro Replay Buffer

    os.makedirs("weights", exist_ok=True)

    for episode in range(n_episodes):
        obs, info = env.reset()
        state = preprocess_state(obs)
        done = False
        total_reward = 0

        while not done:
            # Seleccionar acción (Epsilon-greedy)
            if random.random() > epsilon:
                with torch.no_grad():
                    q_values = policy_net(state.to(device))
                    action = torch.argmax(q_values).item()
            else:
                action = env.action_space.sample()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = preprocess_state(next_obs)

            # GUARDAR EN MEMORIA (en lugar de entrenar directo)
            memory.append((state, action, reward, next_state, done))
            
            state = next_state
            total_reward += reward # type: ignore

            # ENTRENAMIENTO POR BATCH
            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                # Desempaquetamos el batch
                b_states, b_actions, b_rewards, b_next_states, b_dones = zip(*batch)

                b_states = torch.stack(b_states).to(device)
                b_actions = torch.tensor(b_actions).unsqueeze(1).to(device)
                b_rewards = torch.tensor(b_rewards).to(device)
                b_next_states = torch.stack(b_next_states).to(device)
                b_dones = torch.tensor(b_dones, dtype=torch.float32).to(device)

                # Q actual predicho por la policy_net
                current_q = policy_net(b_states).gather(1, b_actions).squeeze()

                # Q futuro predicho por la TARGET_NET (Estabilidad!)
                with torch.no_grad():
                    next_q = target_net(b_next_states).max(1)[0]
                    target_q = b_rewards + (gamma * next_q * (1 - b_dones))

                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Actualizar la red Target cada cierto tiempo
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (episode + 1) % 100 == 0:
            print(f"Episodio {episode+1}: Recompensa Media (últ 100)={total_reward}, Epsilon={epsilon:.3f}")

    torch.save(policy_net.state_dict(), 'weights/modelo_agente_v3.pth')
    print("Entrenamiento completado y modelo guardado.")

if __name__ == "__main__":
    main()