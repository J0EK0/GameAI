import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import tqdm


# === Supervised Dataset ===
class ProlySupervisedDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.data = []
        with open(jsonl_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                obs = entry["obs"]
                action = entry["action"]
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                act_tensor = torch.tensor(action, dtype=torch.float32)
                self.data.append((obs_tensor, act_tensor))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# === Supervised Model ===
class SimpleSupervisedModel(nn.Module):
    def __init__(self, input_dim: int = 11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 預測 x, z move vector
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


# === Training Function ===
def train_model(jsonl_path="supervised_data.jsonl", save_path="supervised_model.pth",
                batch_size=64, epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ProlySupervisedDataset(jsonl_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleSupervisedModel(input_dim=11).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for obs_batch, act_batch in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            pred = model(obs_batch)
            loss = criterion(pred, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss = {epoch_loss:.4f}")

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")


if __name__ == "__main__":
    train_model(
        jsonl_path="supervised_data.jsonl",
        save_path="supervised_model.pth",
        batch_size=64,
        epochs=20,
        lr=1e-3
    )
