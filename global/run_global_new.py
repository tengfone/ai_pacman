import pygame
import numpy as np
import random
from pygame.locals import *
import sys
sys.path.append("./..")
from constants import *
from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup
from walls import WallGroup
from fruit import Fruit
from pauser import Pauser
from levels import LevelController
from text import TextGroup
from sprites import Spritesheet
from maze import Maze
import threading
import time
import os
from agent import *
import matplotlib
matplotlib.use('TKAgg', force=True)
import matplotlib.pyplot as plt
from IPython import display
import math

DANGER_INDEX = 1
PELLET_INDEX = 2
POWERPELLET_INDEX = 3
GHOST_INDEX = 4
WALL_INDEX = 6

global plotResults

class GameController(object):
    pygame.display.set_caption('Pacman')
    # change or delete as necessary, unused for now
    AI_UP = 0
    AI_DOWN = 1
    AI_LEFT = 2
    AI_RIGHT = 3

    directionMapper = {"0_-1": 0, "0_1": 1, "-1_0": 2, "1_0": 3}

    def get_direction(self, state, index, X, Y, wall_multiplier = 1):
        X = int(X/16)
        Y = int(Y/16)
        state[index][X][Y] = 1

    def get_state(self):
        # Up Down Left Right
        # Direction, Danger, Coin, Powerup, Ghost, Cherry, Wall
        state = np.zeros([7, NCOLS, NROWS])
        
        roundedPacmanPositionX = TILEWIDTH * math.ceil(self.pacman.position.x/TILEWIDTH)
        roundedPacmanPositionY = TILEHEIGHT * math.ceil(self.pacman.position.y/TILEHEIGHT)
        index = 0
        self.get_direction(state, index, roundedPacmanPositionX, roundedPacmanPositionY)

        for ghost in self.ghosts:
            index = GHOST_INDEX if ghost.mode.name == "FREIGHT" else DANGER_INDEX
            ghostPositionX = TILEWIDTH * math.ceil(ghost.position.x/TILEWIDTH)
            ghostPositionY = TILEHEIGHT * math.ceil(ghost.position.y/TILEHEIGHT)
            self.get_direction(state, index, ghostPositionX, ghostPositionY)

        for pellet in self.pellets.pelletList:
            index = POWERPELLET_INDEX if pellet.name == "powerpellet" else PELLET_INDEX
            self.get_direction(state, index, pellet.position.x, pellet.position.y)
        
        if self.fruit is not None:
            index = 5
            self.get_direction(state, index, self.fruit.position.x, self.fruit.position.y)
        
        for wall in self.walls.wallList:
            index = WALL_INDEX
            self.get_direction(state, index, wall.position.x, wall.position.y)

        # print(state)
        # state = state.reshape(-1)
        # print(state.tolist())
        return np.array(state, dtype=int)


    # any use?

    # def move(self, action):
    #     if action == self.AI_UP:
    #         # print("Move Up")
    #         keyEvent = pygame.event.Event(pygame.locals.KEYDOWN, key=pygame.locals.K_UP)
    #         pygame.event.post(keyEvent)
    #     elif action == self.AI_DOWN:
    #         # print("Move Down")
    #         keyEvent = pygame.event.Event(pygame.locals.KEYDOWN, key=pygame.locals.K_DOWN)
    #         pygame.event.post(keyEvent)
    #     elif action == self.AI_LEFT:
    #         # print("Move Left")
    #         keyEvent = pygame.event.Event(pygame.locals.KEYDOWN, key=pygame.locals.K_LEFT)
    #         pygame.event.post(keyEvent)
    #     elif action == self.AI_RIGHT:
    #         # print("Move Right")
    #         keyEvent = pygame.event.Event(pygame.locals.KEYDOWN, key=pygame.locals.K_RIGHT)
    #         pygame.event.post(keyEvent)

    def __init__(self):
        pygame.init()
        self.plotted = False
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.background = None
        self.background_flash = None
        self.setBackground()
        self.clock = pygame.time.Clock()
        self.pelletsEaten = 0
        self.fruit = None
        self.pause = Pauser(True)
        self.level = LevelController()
        self.text = TextGroup()
        self.score = 0
        self.gameover = False
        self.sheet = Spritesheet()
        self.maze = Maze(self.sheet)
        self.flashBackground = False
        self.state = 'intro'
        self.running = True
        self.reward = 0
        self.waitForCheck = True

    def introGame(self):
        print("Intro Session")

    def setBackground(self):
        self.background = pygame.surface.Surface(SCREENSIZE).convert()
        self.background_flash = pygame.surface.Surface(SCREENSIZE).convert()
        self.background.fill(BLACK)

    def startGame(self):
        print("Starting game")
        self.level.reset()
        levelmap = self.level.getLevel()
        self.maze.getMaze(levelmap["name"].split(".")[0])
        self.maze.constructMaze(
            self.background, self.background_flash, levelmap["row"])
        self.nodes = NodeGroup(levelmap["name"])
        self.pellets = PelletGroup(levelmap["name"])
        self.walls = WallGroup(levelmap["name"])
        self.pacman = Pacman(self.nodes, self.sheet)
        self.ghosts = GhostGroup(self.nodes, self.sheet)
        self.pelletsEaten = 0
        self.fruit = None
        self.pause.force(True)
        self.text.showReady()
        self.text.updateLevel(self.level.level+1)
        self.gameover = False
        self.maze.reset()
        self.flashBackground = False

    def startAIGame(self):
        print("Starting AI Learning")
        self.level = LevelController('rl')
        self.level.reset()
        levelmap = self.level.getLevel()
        self.maze.getMaze(levelmap["name"].split(".")[0])
        self.maze.constructMaze(
            self.background, self.background_flash, levelmap["row"])
        self.nodes = NodeGroup(levelmap["name"])
        self.pellets = PelletGroup(levelmap["name"])
        self.walls = WallGroup(levelmap["name"])
        self.pacman = Pacman(self.nodes, self.sheet, 'rl')
        self.pacmanPrevPosition = self.pacman.position
        self.ghosts = GhostGroup(self.nodes, self.sheet)
        self.pelletsEaten = 0
        self.fruit = None
        self.pause.force(True)
        self.text.updateLevel(self.level.level+1)
        self.gameover = False
        self.maze.reset()
        self.flashBackground = False

    def startLevel(self):
        print("Start new level")
        levelmap = self.level.getLevel()
        self.setBackground()
        self.maze.getMaze(levelmap["name"].split(".")[0])
        self.maze.constructMaze(
            self.background, self.background_flash, levelmap["row"])
        self.nodes = NodeGroup(levelmap["name"])
        self.pellets = PelletGroup(levelmap["name"])
        self.pacman.nodes = self.nodes
        self.pacman.reset()
        self.pacmanPrevPosition = self.pacman.position
        self.ghosts = GhostGroup(self.nodes, self.sheet)
        self.pelletsEaten = 0
        self.fruit = None
        self.pause.force(True)
        self.text.updateLevel(self.level.level+1)
        self.flashBackground = False
        self.maze.reset()

    def restartLevel(self):
        print("Restart current level")
        if self.state == 'rl':
            self.score = 0
        self.pacman.reset()
        self.ghosts = GhostGroup(self.nodes, self.sheet)
        self.pause.force(True)
        self.fruit = None
        self.flashBackground = False
        self.maze.reset()

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if self.gameover:
                        if self.state == 'rl':
                            self.startAIGame()
                        else:
                            self.startGame()
                    else:
                        self.pause.player()
                        if self.pause.paused:
                            self.text.showPause()
                        else:
                            self.text.hideMessages()
                elif event.key == K_UP:
                    self.pacman.setDirection(UP)
                elif event.key == K_DOWN:
                    self.pacman.setDirection(DOWN)
                elif event.key == K_LEFT:
                    self.pacman.setDirection(LEFT)
                elif event.key == K_RIGHT:
                    self.pacman.setDirection(RIGHT)

    def checkPelletEvents(self):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            # REWARDS FOR COINS
            self.pelletsEaten += 1
            self.score += pellet.points
            if (self.pelletsEaten == 70 or self.pelletsEaten == 140):
                if self.fruit is None:
                    levelmap = self.level.getLevel()
                    self.fruit = Fruit(
                        self.nodes, self.sheet, levelmap["fruit"])
            self.pellets.pelletList.remove(pellet)
            if pellet.name == "powerpellet":
                self.ghosts.resetPoints()
                self.ghosts.freightMode()
            if self.pellets.isEmpty():
                self.pacman.visible = False
                self.ghosts.hide()
                self.pause.startTimer(3, "clear")
                self.flashBackground = True

    def checkGhostEvents(self):
        self.ghosts.release(self.pelletsEaten)
        ghost = self.pacman.eatGhost(self.ghosts)
        if ghost is not None:
            if ghost.mode.name == "FREIGHT":
                self.score += ghost.points
                self.text.createTemp(ghost.points, ghost.position)
                self.ghosts.updatePoints()
                ghost.spawnMode(speed=2)
                self.pause.startTimer(1)
                self.pacman.visible = False
                ghost.visible = False

            elif ghost.mode.name != "SPAWN":
                self.pacman.loseLife()
                self.ghosts.hide()
                self.pause.startTimer(3, "die")

    def checkFruitEvents(self):
        if self.fruit is not None:
            if self.pacman.eatFruit(self.fruit):
                self.score += self.fruit.points
                self.text.createTemp(self.fruit.points, self.fruit.position)
                self.fruit = None

            elif self.fruit.destroy:
                self.fruit = None

    def resolveDeath(self):
        if self.pacman.lives == 0:
            self.gameover = True
            self.plotted = False
            self.pacman.visible = False
            self.text.showGameOver()
        else:
            self.restartLevel()
        self.pause.pauseType = None

    def resolveLevelClear(self):
        self.level.nextLevel()
        self.startLevel()
        self.pause.pauseType = None

    def render(self):
        self.screen.blit(self.maze.background, (0, 0))
        # self.nodes.render(self.screen)
        self.pellets.render(self.screen)
        if self.fruit is not None:
            self.fruit.render(self.screen)
        self.pacman.render(self.screen)
        self.ghosts.render(self.screen)
        self.pacman.renderLives(self.screen)
        self.text.render(self.screen)
        pygame.display.update()

    ################## intro FUNCTIONS ##################

    def start_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.state = 'normal'
                self.startGame()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                self.state = 'rl'
                self.startAIGame()

    def start_draw(self):
        self.screen.fill((0, 0, 0))
        self.draw_text('PUSH SPACE BAR TO PLAY', self.screen, [SCREENWIDTH//2, SCREENHEIGHT//2 - 50],
                       14, (255, 255, 51), centered=True)
        self.draw_text('PUSH A TO ENTER AI MODE', self.screen, [SCREENWIDTH//2, SCREENHEIGHT//2 - 0],
                       14, (255, 255, 51), centered=True)
        self.draw_text('1 PLAYER ONLY', self.screen, [SCREENWIDTH//2, SCREENHEIGHT//2 + 50],
                       14, (44, 167, 198), centered=True)
        pygame.display.update()

    def draw_text(self, text, screen, pos, size, color, centered=False):
        font = pygame.font.Font(FONTTYPE, size)
        text = font.render(text, False, color)
        text_size = text.get_size()
        if centered:
            pos[0] = pos[0] - text_size[0] // 2
            pos[1] = pos[1] - text_size[1] // 2
        screen.blit(text, pos)

    def run(self):
        
        while self.running:
            if self.state == 'intro':
                self.start_events()
                self.start_draw()

            if self.state == 'normal':
                if not self.gameover:
                    dt = self.clock.tick(30) / 1000.0
                    if not self.pause.paused:
                        self.pacman.update(dt)
                        self.ghosts.update(dt, self.pacman)
                        if self.fruit is not None:
                            self.fruit.update(dt)

                        if self.pause.pauseType != None:
                            self.pause.settlePause(self)

                        self.checkPelletEvents()
                        self.checkGhostEvents()
                        self.checkFruitEvents()

                    else:
                        if self.flashBackground:
                            self.maze.flash(dt)

                        if self.pacman.animateDeath:
                            self.pacman.updateDeath(dt)

                    self.pause.update(dt)
                    self.pellets.update(dt)
                    self.text.update(dt)
                self.checkEvents()
                self.text.updateScore(self.score)
                self.render()

            if self.state == 'rl':
                if not self.gameover:
                    dt = self.clock.tick(30) / 1000.0
                    if not self.pause.paused:
                        self.pacman.updateAI(dt)
                        self.ghosts.update(dt, self.pacman)
                        if self.fruit is not None:
                            self.fruit.update(dt)

                        if self.pause.pauseType != None:
                            self.pause.settlePause(self)
                        self.checkPelletEvents()
                        self.checkGhostEvents()
                        self.checkFruitEvents()
                        self.waitForCheck = False

                    else:
                        if self.flashBackground:
                            self.maze.flash(dt)

                        if self.pacman.animateDeath:
                            self.pacman.updateDeath(dt)

                    self.pause.update(dt)
                    self.pellets.update(dt)
                    self.text.update(dt)
                else:
                    global plotResults

                    try:
                        if not self.plotted:
                            plot(plotResults[0], plotResults[1])
                            self.plotted = True
                    except NameError:
                        pass
                
                self.checkEvents()
                self.text.updateScore(self.score)
                self.render()

        pygame.quit()
        sys.exit()

def play_step(action, old_state):
    if game.pause.paused:
        keyEvent = pygame.event.Event(
            pygame.locals.KEYDOWN, key=pygame.locals.K_SPACE)
        pygame.event.post(keyEvent)
        time.sleep(0.1)
    temp_action = max(action)
    actionIndex = action.index(temp_action)

    if actionIndex == 0:
        # print("Move Up")
        keyEvent = pygame.event.Event(
            pygame.locals.KEYDOWN, key=pygame.locals.K_UP)
        pygame.event.post(keyEvent)
    elif actionIndex == 1:
        # print("Move Down")
        keyEvent = pygame.event.Event(
            pygame.locals.KEYDOWN, key=pygame.locals.K_DOWN)
        pygame.event.post(keyEvent)
    elif actionIndex == 2:
        # print("Move Left")
        keyEvent = pygame.event.Event(
            pygame.locals.KEYDOWN, key=pygame.locals.K_LEFT)
        pygame.event.post(keyEvent)
    elif actionIndex == 3:
        # print("Move Right")
        keyEvent = pygame.event.Event(
            pygame.locals.KEYDOWN, key=pygame.locals.K_RIGHT)
        pygame.event.post(keyEvent)
    
    while game.waitForCheck:
        if game.pause.paused:
            keyEvent = pygame.event.Event(
                pygame.locals.KEYDOWN, key=pygame.locals.K_SPACE)
            pygame.event.post(keyEvent)
            time.sleep(0.1)

    game.waitForCheck = True

    actionMapperX = {0: 0, 1: 0, 2: -1, 3: 1}
    actionMapperY = {0: -1, 1: 1, 2: 0, 3: 0}

    if game.pacmanPrevPosition == game.pacman.position:
        game.reward -= 50
    game.pacmanPrevPosition = game.pacman.position

    # if old_state[8*WALL_INDEX+actionMapper[actionIndex]] > 0.9:
    #     game.reward -= 50
    # elif old_state[8*WALL_INDEX+actionMapper2[actionIndex][0]] > 0.9 or old_state[8*WALL_INDEX+actionMapper2[actionIndex][1]] > 0.9:
    #     game.reward -= 10

    # Check GameOver
    gameover = game.gameover
    pacmanIndex = np.where(old_state[0] == 1)
    print(pacmanIndex)
    pacmanIndexX = pacmanIndex[0][0] + actionMapperX[actionIndex]
    pacmanIndexY = pacmanIndex[1][0] + actionMapperY[actionIndex]
    
    if old_state[DANGER_INDEX, pacmanIndexX, pacmanIndexY] > 0.9:
        game.reward -= 50
        if gameover:
            game.reward = -200
    # elif (old_state[DANGER_INDEX, actionMapper2[actionIndex][0]] > 0.9 or old_state[DANGER_INDEX, actionMapper2[actionIndex][1]] > 0.9) and old_state[0][actionMapper[actionIndex]] < 0.9:
    #     game.reward -= 10
    
    if old_state[GHOST_INDEX, pacmanIndexX, pacmanIndexY] > 0.9:
        game.reward += 200
    # elif (old_state[GHOST_INDEX, actionMapper2[actionIndex][0]] > 0.9 or old_state[GHOST_INDEX, actionMapper2[actionIndex][1]] > 0.9) and old_state[0][actionMapper[actionIndex]] < 0.9:
    #     game.reward += 50
    
    if old_state[PELLET_INDEX, pacmanIndexX, pacmanIndexY] > 0.9:
        game.reward += 10
    # elif (old_state[PELLET_INDEX, actionMapper2[actionIndex][0]] > 0.9 or old_state[PELLET_INDEX, actionMapper2[actionIndex][1]] > 0.9) and old_state[0][actionMapper[actionIndex]] < 0.9:
    #     game.reward += 5

    if old_state[POWERPELLET_INDEX, pacmanIndexX, pacmanIndexY] > 0.9:
        game.reward += 20
    # elif (old_state[POWERPELLET_INDEX, actionMapper2[actionIndex][0]] > 0.9 or old_state[POWERPELLET_INDEX, actionMapper2[actionIndex][1]] > 0.9) and old_state[0][actionMapper[actionIndex]] < 0.9:
    #     game.reward += 10

    print(actionIndex, game.reward)
    return game.reward, gameover, game.score


def trainAI():
    gameReady = False
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    n_games_session = 0
    agent = Agent(load=True)
    while not gameReady:
        try:
            print(game.state)
            gameReady = True
        except NameError:
            print("Waiting for game..")
            time.sleep(0.1)
    hasPacman = False
    while not hasPacman:
        try:
            print(game.pacman)
            hasPacman = True
        except AttributeError:
            print("Waiting for pacman..")
            time.sleep(0.1)

    # TRAINING LOOP
    if game.state == "rl":
        while True:
            game.reward = 0
            old_state = game.get_state()
            roundedPacmanPositionX = TILEWIDTH * math.ceil(game.pacman.position.x/TILEWIDTH)
            roundedPacmanPositionY = TILEHEIGHT * math.ceil(game.pacman.position.y/TILEHEIGHT)
            final_move = agent.get_action(old_state)
            reward, done, score = play_step(final_move, old_state)
            start_time = time.time()
            while time.time() < start_time + 0.5 and not game.gameover and TILEWIDTH * math.ceil(game.pacman.position.x/TILEWIDTH) == roundedPacmanPositionX and TILEHEIGHT * math.ceil(game.pacman.position.y/TILEHEIGHT) == roundedPacmanPositionY:
                continue
            new_state = game.get_state()

            # Train short memory
            agent.train_short_memory(
                old_state, final_move, reward, new_state, done)

            # Remember
            agent.remember(old_state, final_move, reward, new_state, done)

            if done:
                # Train long memory (replay memory)
                game.restartLevel()
                time.sleep(3)
                keyEvent = pygame.event.Event(
                    pygame.locals.KEYDOWN, key=pygame.locals.K_SPACE)
                pygame.event.post(keyEvent)
                agent.n_games += 1
                n_games_session += 1
                agent.train_long_memory()

                if score > agent.record or agent.n_games % 5 == 0:
                    print("Saving model")
                    if score > agent.record:
                        print("Saving model with highscore")
                        agent.record = score
                    agent.trainer.save(agent.n_games, agent.record)

                # Plotting Graph
                print(f"Game {agent.n_games}, Score: {score}, Record: {agent.record}")

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / n_games_session
                plot_mean_scores.append(mean_score)

                global plotResults

                plotResults = [plot_scores, plot_mean_scores]
                
                time.sleep(3)
                keyEvent = pygame.event.Event(
                    pygame.locals.KEYDOWN, key=pygame.locals.K_SPACE)
                pygame.event.post(keyEvent)
    else:
        while True:
            print(game.get_state())
            time.sleep(1)
        # while True:
        #     action = random.randint(0, 3)
        #     if action == 0:
        #         print("Move Up")
        #         keyEvent = pygame.event.Event(pygame.locals.KEYDOWN, key=pygame.locals.K_UP)
        #         pygame.event.post(keyEvent)
        #     elif action == 1:
        #         print("Move Down")
        #         keyEvent = pygame.event.Event(pygame.locals.KEYDOWN, key=pygame.locals.K_DOWN)
        #         pygame.event.post(keyEvent)
        #     elif action == 2:
        #         print("Move Left")
        #         keyEvent = pygame.event.Event(pygame.locals.KEYDOWN, key=pygame.locals.K_LEFT)
        #         pygame.event.post(keyEvent)
        #     elif action == 3:
        #         print("Move Right")
        #         keyEvent = pygame.event.Event(pygame.locals.KEYDOWN, key=pygame.locals.K_RIGHT)
        #         pygame.event.post(keyEvent)
        #     print(game.score)
        #     print(game.get_state_closest())
        #     time.sleep(1)


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


global game

def continue_pygame_loop():
    pygame.mainloop(0.1)
    yield

if __name__ == "__main__":
    threads = []

    # threads.append(threading.Thread(target=runGame, daemon=True))
    global game
    
    threads.append(threading.Thread(target=trainAI, daemon=True))

    for thread in threads:
        thread.start()
    game = GameController()
    game.run()

    while True:
        try:
            [t.join(1) for t in threads]
        except KeyboardInterrupt:
            game.running = False
            os._exit(1)