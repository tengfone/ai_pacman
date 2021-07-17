import pygame
from constants import *

class Spritesheet(object):
    def __init__(self):
        self.sheet = pygame.image.load("spritesheet.png").convert()
        transcolor = self.sheet.get_at((0,0))
        print(transcolor)
        self.sheet.set_colorkey((0,0,0,255))
        self.sheet = pygame.transform.scale(self.sheet, (22*TILEWIDTH, 23*TILEHEIGHT))
        
    def getImage(self, x, y, width, height):
        x *= width
        y *= height
        self.sheet.set_clip(pygame.Rect(x, y, width, height))
        return self.sheet.subsurface(self.sheet.get_clip())
