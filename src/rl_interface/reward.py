from builtins import property
import math

class Reward:
    def __init__(self,reward):
        self._reward=reward
    
    @property
    def reward(self):
        return self._reward
    
    @reward.setter
    def reward(self,value):
        self._reward = value
    
    def log_reward(self):
        return math.log(abs(self._reward),2)