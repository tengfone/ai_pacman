import pygame
from vector import Vector2
from constants import *

class Wall(object):
    def __init__(self, x, y):
        self.name = "wall"
        self.position = Vector2(x, y)

class WallGroup(object):
    def __init__(self, mazefile):
        self.wallList = []
        self.wallSymbol = ["."]
        self.createWallList(mazefile)
        
    def createWallList(self, mazefile):
        grid = self.readMazeFile(mazefile)
        rows = len(grid)
        cols = len(grid[0])
        for row in range(rows):
            for col in range(cols):
                if (grid[row][col] in self.wallSymbol):
                    self.wallList.append(Wall(col*TILEWIDTH, row*TILEHEIGHT))
        for wall in self.wallList:
            print(wall.position)
                    
    def readMazeFile(self, textfile):
        f = open(textfile, "r")
        lines = [line.rstrip('\n') for line in f]
        return [line.split(' ') for line in lines]