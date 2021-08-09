import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
import os


class LinearQNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=2, stride=2)
        self.linear1 = nn.Linear(1008, 64)
        self.head = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.linear1(x.view(x.size(0), -1)))
        return self.head(x)

class TrainerQ:
    def __init__(self, model, lr, gamma, load) -> None:
        self.model = model
        self.gamma = gamma
        self.lr = lr
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.steps = 0
        if load:
            self.load()

    def save(self, n_games, record, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save({
            'steps': self.steps,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            'n_games': n_games,
            'record': record
            }, file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        if os.path.exists(model_folder_path):
            file_name = os.path.join(model_folder_path, file_name)
            checkpoint = torch.load(file_name)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.steps = checkpoint['steps']
            self.loss = checkpoint['loss']
            return checkpoint['n_games'], checkpoint['record']
        else:
            return 0, 0

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if type(done) is bool:
            state = state.unsqueeze(0).unsqueeze(0)
            next_state = next_state.unsqueeze(0).unsqueeze(0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        elif type(done) is tuple:
            state = state.unsqueeze(1)
            next_state = next_state.unsqueeze(1)

        pred = self.model(state)

        # r + y * max(next_pred Q Value)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * \
                    torch.max(self.model(next_state))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        self.loss = self.criterion(target, pred)
        print("loss", self.loss)
        self.loss.backward()

        self.optimizer.step()
        self.steps += 1

model = LinearQNet()