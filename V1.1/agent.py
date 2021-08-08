import torch
import random
import numpy as np
from run import GameController
from collections import deque
from constants import *
from model import LinearQNet, TrainerQ
import matplotlib.pyplot as plt
from IPython import display

MAX_MEMORY = 1_000_000
BATCH_SIZE = 500
LR = 0.0003
plt.ion()


class Agent:
    def __init__(self, load) -> None:
        self.n_games = 0
        self.record = 0
        self.epsilon = 100/(self.n_games+100)  # Randomness
        self.gamma = 0.8  # Discount rate Must be < 1
        self.memory = deque(maxlen=MAX_MEMORY)  # Popleft if over max mem
        # 28 is number of features, 4 is up,down,left,right
        self.model = LinearQNet(8, 7, 256, 4)
        self.trainer = TrainerQ(self.model, lr=LR, gamma=self.gamma, load = load)
        self.prevPrediction = None
        if load:
            self.n_games, self.record = self.trainer.load()

    def remember(self, state, action, reward, next_state, done):
        # popleft if MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(
                self.memory, BATCH_SIZE)  # Rt a list of tuple
        else:
            mini_sample = self.memory

        # Put together
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random Moves: tradeoff between exploration | exploitation
        #self.epsilon = 0
        self.epsilon = 25/(self.n_games+25)
        final_move = [0, 0, 0, 0]
        if self.n_games < 400 and self.epsilon > random.random():
            move = random.randint(0, 3)
            final_move[move] = 1
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            prediction = self.model(state0)
            print("random", prediction)
        else:
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            prediction = self.model(state0)
            # if self.prevPrediction is None:
            #     self.prevPrediction = prediction
            # print(prediction - self.prevPrediction)
            #move = torch.argmax(prediction - self.prevPrediction).item()  # get best move [0,0,1,0]
            # self.prevPrediction = prediction
            print("predict", prediction)
            move = torch.argmax(prediction).item()  # get best move [0,0,1,0]
            final_move[move] = 1

        return final_move
