import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class LinearQNet(nn.Module):
    def __init__(self, w, h, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(56, 512)
        self.linear2 = nn.Linear(512, 64)
        # self.linear1 = nn.Linear(7056, 1028)
        # self.linear2 = nn.Linear(1028, 64)
        # self.conv1 = nn.Conv2d(7, 4, kernel_size=1, stride=1)
        # self.conv2 = nn.Conv2d(4, 1, kernel_size=1, stride=1)
        # self.bn1 = nn.BatchNorm2d(1)
        # self.conv2 = nn.Conv2d(2, 4, kernel_size=1, stride=1)
        # self.bn2 = nn.BatchNorm2d(4)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        # def conv2d_size_out(size, kernel_size = 1, stride = 1):
        #     return (size - (kernel_size - 1) - 1) // stride  + 1
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        self.head = nn.Linear(64, output_size)

    # def forward(self, x):
    #     x = F.relu(self.linear1(x))
    #     x = self.linear2(x)
    #     return x
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # x = F.relu(self.bn2(self.conv2(x)))
        return self.head(x.view(x.size(0), -1))

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
            # self.model.load_state_dict(torch.load(file_name))
            # print("model loaded")
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

        # if len(state.shape) == 1:
        # if len(state.shape) == 2:
        #     state = state.unsqueeze(0).unsqueeze(0)
        #     next_state = next_state.unsqueeze(0).unsqueeze(0)
        #     action = torch.unsqueeze(action, 0)
        #     reward = torch.unsqueeze(reward, 0)
        #     done = (done, )
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        print(state)

        # Predict Q values with current state
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
