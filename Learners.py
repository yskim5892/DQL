import numpy as np
from Bauhausbreak import check_matching

def encode_block(block):
    color = block // 9
    shape = (block % 9) // 3
    pattern = block % 3
    result = np.zeros(9)
    result[color] = 1
    result[shape + 3] = 1
    result[pattern + 6] = 1
    return result

def two_block_match_bits(b1, b2):
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

def drop_pos(blocks, x):
    for y in range(blocks.shape[1]):
        if(blocks[x][y] == 27):
            return y
    return None

def process_state_BHB(hwc, state):
    blocks = np.zeros(hwc)
    for x in range(hwc[1]):
        for y in range(hwc[0]):
            if(state.blocks[x][y] < 27):
                blocks[y][x][0:9] = encode_block(state.blocks[x][y])
            else:
                blocks[y][x][9] = state.blocks[x][y] - 27

            i = 0
            for dx, dy in [(-2, 0), (-1, 0), (1, 0),(2, 0), (0, -1)]:
                try:
                    blocks[y][x][10+i] = check_matching(state.blocks[x][y], state.blocks[x+dx][y+dy], state.current_block) * 100
                except IndexError:
                    pass
                i += 1

            '''for dx, dy in [(-2, 0), (-1, 0), (1, 0), (2, 0), (0, -1), (0, -2)]:
                try:
                    blocks[y][x][13+3*i:16+3*i] = self.two_block_match_bits(state.blocks[x][y], state.blocks[x+dx][y+dy])
                except IndexError:
                    pass
                i += 1'''

        drop_y = drop_pos(state.blocks, x)
        if drop_y != None:
            blocks[drop_y][x][15] = 100

    blocks = np.reshape(blocks, [hwc[0] * hwc[1] * hwc[2]])
    current_block = encode_block(state.current_block)

    return np.concatenate((blocks, current_block, [state.gauge], [int(state.is_terminal)]), 0)


def decode_action_BHB(action):
    return action
def is_success_BHB(state, reward):
    return state.grade >= 3



def process_state_RT(hwc, env, state):
    track = np.zeros(hwc)

    for x in range(hwc[1]):
        for y in range(hwc[0]):
            if(state.track[y][x] == 1):
                track[y][x][0] = 1

    if not env.out_of_track(int(state.x[0]), int(state.x[1])):
        track[int(state.x[0])][int(state.x[1])][1] = 1
    track[int(state.dest[0])][int(state.dest[1])][2] = 1
    
    track = np.reshape(track, [hwc[0] * hwc[1] * hwc[2]])

    return np.concatenate([track, state.x, state.dest, [int(state.is_terminal)]], 0)

def decode_action_RT(action):
    return [action // 3 - 1, action % 3 - 1]
def is_success_RT(state, reward):
    return reward > 0
