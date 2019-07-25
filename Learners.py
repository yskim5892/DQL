from DQLearner import DQLearner
import numpy as np

class BHBLearner(DQLearner):
    def process_state(self, state):
        blocks = np.zeros(self.hwc)
        for y in range(self.h):
            for x in range(self.w):
                blocks[y][x][state.blocks[x][y]] = 1

        blocks = np.reshape(blocks, [self.h * self.w * self.c])

        current_block = np.zeros([27])
        current_block[state.current_block] = 1

        return np.concatenate((blocks, current_block, [state.gauge], [int(state.is_terminal)]), 0)

    def decode_action(self, action):
        return action

class RTLearner(DQLearner):
    def process_state(self, state):
        track = np.zeros([self.h, self.w, self.c])

        for x in range(self.w):
            for y in range(self.h):
                track[x][y][state.track[y][x]] = 1

        if not self.env.out_of_track(int(state.x[0]), int(state.x[1])):
            track[int(state.x[0])][int(state.x[1])] = [0, 0, 1, 0]
        track[int(state.dest[0])][int(state.dest[1])] = [0, 0, 0, 1]
        
        track = np.reshape(track, [self.h * self.w * self.c])

        return np.concatenate([track, state.v, state.x, state.dest, [int(state.is_terminal)]], 0)
    
    def decode_action(self, action):
        return [action // 3 - 1, action % 3 - 1]
