import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory=[]
        self.capacity=capacity

    def getMemory(self):
        return self.memory

    def getCapacity(self):
        return self.capacity

    def add(self, state, action, reward, nextState, done):
        if len(self.memory) > self.capacity:
            return
        experience = (self, state, action, reward, nextState, done)
        self.memory.append(experience)

    def sample(self, batch_size):
        pass

