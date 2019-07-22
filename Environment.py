import numpy as np

class Action:
    def __init__(self):
        pass

class State:
    def __init__(self, is_terminal):
        self.is_terminal = is_terminal

class Response:
    def __init__(self, prob, state, reward):
        self.prob = prob
        self.state = state
        self.reward = reward

class Environment:
    def __init__(self, states, init_state, gamma):
        self.states = states
        self.state = init_state
        self.gamma = gamma

    def respond(self, action):
        pass

    def possible_actions(self, state):
        pass
