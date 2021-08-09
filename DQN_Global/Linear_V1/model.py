import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class LinearQNet(nn.Module):
    def __init__(self, w, h, hidden_size, output_size):
        super().__init__()
<<<<<<< HEAD
        self.linear1 = nn.Linear(7056, 1028)
        self.linear2 = nn.Linear(1028, 64)
=======
        self.linear1 = nn.Linear(1008, 256)
        self.linear2 = nn.Linear(256, 64)
>>>>>>> fc13320b078911dfbbb3cfdad844e54c52c306f5
        self.head = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
<<<<<<< HEAD
        return self.head(x.view(x.size(0), -1))
=======
        return self.head(x)
>>>>>>> fc13320b078911dfbbb3cfdad844e54c52c306f5

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
<<<<<<< HEAD
    
=======

>>>>>>> fc13320b078911dfbbb3cfdad844e54c52c306f5
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
<<<<<<< HEAD
    
=======

>>>>>>> fc13320b078911dfbbb3cfdad844e54c52c306f5
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

<<<<<<< HEAD
        if len(state.shape) == 1:
=======
        if type(done) is bool:
>>>>>>> fc13320b078911dfbbb3cfdad844e54c52c306f5
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

<<<<<<< HEAD
        # Predict Q values with current state
=======
>>>>>>> fc13320b078911dfbbb3cfdad844e54c52c306f5
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
<<<<<<< HEAD
=======
        print("loss", self.loss)
>>>>>>> fc13320b078911dfbbb3cfdad844e54c52c306f5
        self.loss.backward()

        self.optimizer.step()
        self.steps += 1
