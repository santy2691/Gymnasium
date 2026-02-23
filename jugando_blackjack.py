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
        logits = self.linear_relu_stack(x)
        return logits

def preprocess_state(state):
    return torch.tensor([state[0], state[1], float(state[2])], dtype=torch.float32)


model = Agente()
loaded = torch.load("./weights/modelo_agente_v3.pth", map_location="cpu")
if isinstance(loaded, dict):
    model.load_state_dict(loaded)
else:
    model = loaded
model.eval()

state = [20, 10, 1]
input_t = preprocess_state(state).unsqueeze(0)
print(input_t)

with torch.no_grad():
    logits = model(input_t)
    probs = torch.softmax(logits, dim=1)
    action = torch.argmax(probs, dim=1).item()

logits_np = logits.cpu().detach().numpy()
probs_np = probs.cpu().detach().numpy()

print("Logits:", logits_np)
print("Probabilidades:", probs_np)
print("Acción elegida:", action)