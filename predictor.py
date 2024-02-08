import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class CustomDS(Dataset):
    def __init__(self, simulation_data):
        # Flatten the observations and actions. We have 1 datapoint per step per agent
        observations = torch.tensor(simulation_data["observations"], dtype=torch.float32)
        observations = observations.view(-1, observations.shape[-1])
        actions = torch.tensor(simulation_data["actions"], dtype=torch.float32)
        actions = actions.view(-1, actions.shape[-1])
        rewards = torch.tensor(simulation_data["rewards"], dtype=torch.float32)
        rewards = rewards.view(-1, 1)

        self.x = torch.cat([observations, actions], dim=1)
        self.y = rewards

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class Predictor():

    def __init__(self, in_size, hidden_size=32):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1)
        )

    def train_model(self, simulation_data, epochs, batch_size, lr):
        ds = CustomDS(simulation_data)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        average_train_mse = 0
        self.model.train()
        for epoch in range(epochs):
            for x, y in dl:
                optimizer.zero_grad()
                y_pred = self.model(x)
                loss = loss_fn(y_pred, y)
                average_train_mse += loss.item() * batch_size
                loss.backward()
                optimizer.step()

        print(f"Average train MSE: {average_train_mse / len(ds)}")

        self.model.eval()

    def predict(self, observations, actions):
        obs = torch.tensor(observations, dtype=torch.float32)
        act = torch.tensor(actions, dtype=torch.float32)
        rewards = self.model(torch.cat([obs, act], dim=1))
        return rewards.tolist()
    

class RFPredictor():
    def __init__(self):
        self.model = RandomForestRegressor(n_jobs=-1, max_features="sqrt")
    
    def train_model(self, simulation_data):
        observations = simulation_data["observations"]
        obs = observations.reshape(-1, observations.shape[-1])
        actions = simulation_data["actions"]
        act = actions.reshape(-1, actions.shape[-1])

        x = np.concatenate([obs, act], axis=1)
        y = simulation_data["rewards"].reshape(-1)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
        self.model.fit(x_train, y_train)
        print("validation mse: ", mean_squared_error(y_val, self.model.predict(x_val)))

    def predict(self, observations, actions):
        obs = observations.reshape(-1, observations.shape[-1])
        act = actions.reshape(-1, actions.shape[-1])
        x = np.concatenate([obs, act], axis=1)
        return self.model.predict(x)