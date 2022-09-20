from builtins import property


class Reward:
    def __init__(self):
        self._reward=0
    
    @property
    def reward(self):
        return self._reward
    
    @x.setter
    def reward(self,value):
        self._reward = value
    
    