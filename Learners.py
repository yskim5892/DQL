from DQLearner import DQLearner
import numpy as np

class BHBLearner(DQLearner):
    def process_state(self, state):
        blocks = np.zeros(self.hwc)
        for x in range(self.w):
            for y in range(self.h):
                blocks[y][x][state.blocks[x][y]] = 1
                blocks[y][x][31:34] = self.two_block_match_bits(state.blocks[x][y], state.current_block)
                
                i = 0
                for dx, dy in [(-2, 0), (-1, 0), (1, 0), (2, 0), (0, -1), (0, -2)]:
                    try:
                        blocks[y][x][34+3*i:37+3*i] = self.two_block_match_bits(state.blocks[x][y], state.blocks[x+dx][y+dy])
                    except IndexError:
                        pass
                    i += 1

            drop_pos = self.drop_pos(state.blocks, x)
            if drop_pos != None:
                blocks[drop_pos][x][52] = 1

        blocks = np.reshape(blocks, [self.h * self.w * self.c])

        current_block = np.zeros([27])
        current_block[state.current_block] = 1

        return np.concatenate((blocks, current_block, [state.score], [state.gauge], [int(state.is_terminal)]), 0)

    def two_block_match_bits(self, b1, b2):
        result = np.zeros(3)
        if(b1 >= 27 or b2 >= 27):
            return result

        if(b1 // 9 == b2 // 9):
            result[0] = 1
        if((b1 % 9) // 3 == (b2 % 9) // 3):
            result[1] = 1
        if(b1 % 3 == b2 % 3):
            result[2] = 1
        return result

    def drop_pos(self, blocks, x):
        for y in range(self.h):
            if(blocks[x][y] == 27):
                return y
        return None

    def decode_action(self, action):
        return action
    def is_success(self, reward):
        return reward > 200

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
    def is_success(self, reward):
        return reward > 0
