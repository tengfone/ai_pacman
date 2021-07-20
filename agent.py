import torch
import random
import numpy as np
from run import GameController
from collections import deque
from constants import *

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0  # Randomness
        self.gamma = 0  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # Popleft if over max mem
        self.model = None
        self.trainer = None

    def get_state(self, game):

        # TODO: Implement
        directionLeft = LEFT
        directionUp = UP
        directionDown = DOWN
        directionRight = RIGHT

        state = [
            # Move Direction

            # Danger

            # Power Up Location
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE :
            mini_sample = random.sample(self.memory, BATCH_SIZE) # Rt a list of tuple
        else:
            mini_sample = self.memory

        # Put together
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random Moves: tradeoff between exploration | exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state,dtype=torch.float)
            prediction = self.model.predict(state0)
            move = torch.argmax(prediction).item() # get best move [0,0,1,0]
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = GameController()
    while True:
        # Get old state
        old_state = agent.get_state(game)

        # Get move
        final_move = agent.get_action(old_state)

        # Perform move and get new state, TODO: implement play_step
        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(
            old_state, final_move, reward, new_state, done)

        # Remember
        agent.remember(old_state, final_move, reward, new_state, done)

        if done:
            # Train long memory (replay memory)
            game.restartLevel()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # TODO: agent.model.save()

            print(f"Game {agent.n_games}, Score: {score}, Record: {record}")


if __name__ == '__main__':
    train()
