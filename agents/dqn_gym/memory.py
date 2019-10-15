# import numpy as np
# from collections import deque
#
#
# class Memory:
#     def __init__(self, max_size=1000):
#         self.buffer = deque(maxlen=max_size)
#
#     def add(self, experience):
#         self.buffer.append(experience)
#
#     def sample(self, batch_size):
#         idx = np.random.choice(np.arange(len(self.buffer)),
#                                size=batch_size,
#                                replace=False)
#         return [self.buffer[ii] for ii in idx]


import random


class Memory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def add(self, experience):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)