class LevelController(object):
    def __init__(self,mode='normal'):
        if mode == 'normal':
            self.level = 0
            self.levelmaps = {0: {"name":"maze1.txt", "row":0, "fruit":"cherry"},
                            1: {"name":"maze2.txt", "row":1, "fruit":"banana"},
                            2: {"name":"maze3.txt", "row":2, "fruit":"apple"}}
        
        if mode == 'rl':
            self.level = 0
            self.levelmaps = {0: {"name":"maze1.txt", "row":0, "fruit":"cherry"},
                1: {"name":"maze2.txt", "row":1, "fruit":"banana"}}
        
    def nextLevel(self):
        self.level += 1

    def reset(self):
        self.level = 0

    def getLevel(self):
        return self.levelmaps[self.level % len(self.levelmaps)]
