import random


class Memory(object):

    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, experience):
        """Saves a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def set_capacity(self, capacity):
        self.capacity = capacity
