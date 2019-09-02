from Environment import *
from utils import *
from DQLearner import Record
import numpy as np
import json

class RT_State(State):
    def __init__(self, is_terminal, track, x, v, dest):
        self.is_terminal = is_terminal
        self.track = track
        self.x = x
        self.v = v
        self.dest = dest

class RT_Environment(Environment):
    def __init__(self, size, track = None):
        self.size = size

        if(track is None):
            self.construct_track()
        else:
            self.track_poses = []
            self.track = track
            for x in range(self.size):
                for y in range(self.size):
                    if(track[y][x] == 1):
                        self.track_poses.append([y, x])
        self.initialize_environment()

    def make_success_records(self, size, n, gamma, process_state, decode_action, process_action):
        records = []
        for _ in range(size):
            self.initialize_environment()
            x_history = [self.dest]
            action_history = []
            G = 0
            for i in range(n):
                if(i == 0):
                    G = 100
                else:
                    G = gamma * G - 1
                x = x_history[i]
                for action in range(0, 9):
                    x_ = decode_action(action)

                    new_x = (x[0] + x_[0], x[1] + x_[1])
                    if(self.out_of_track(new_x[0], new_x[1]) or self.track[new_x[0]][new_x[1]] == 0 or (new_x[0] == self.dest[0] and new_x[1] == self.dest[1])):
                        continue
                    else:
                        action_history.append(process_action(8 - action))
                        break
                x_history.append(new_x)
                state0 = RT_State(False, self.track, new_x, (0, 0), self.dest)
                state1 = RT_State(True, self.track, self.dest, (0, 0), self.dest)
                record = Record(process_state(state0), action_history[i], G, process_state(state1))
                records.append(record)

        return records
            
    
    def construct_track(self):
        n = self.size
        track = np.zeros([n, n], dtype=np.dtype(int)) # 0 : empty, 1 : track

        prev_x_min = 0
        prev_x_max = n - 1
        self.track_poses = []
        for y in range(0, n):
            x = 0
            w = n
            while(True):
                x = np.random.randint(0, n - 2)
                w = np.random.randint(1, n - x + 1)
                if(not (x > prev_x_max or x + w - 1 < prev_x_min)):
                    break
            prev_x_min = x
            prev_x_max = x + w - 1
            for i in range(x, x+w):
                track[y][i] = 1
                self.track_poses.append([y, i])
       
        ind = np.random.choice(range(len(self.track_poses)))
        self.dest = self.track_poses[ind]

        self.track = track

    def initialize_environment(self):
        ind = np.random.choice(range(len(self.track_poses)), 2)
        x = self.track_poses[ind[0]]
        self.dest = self.track_poses[ind[1]]
        self.state = RT_State(False, self.track, x, [0, 0], self.dest)

    def out_of_track(self, y, x):
        return y < 0 or y >= self.size or x < 0 or x >= self.size or self.track[y][x] == 0

    def respond(self, action):
        y = self.state.x[0]
        x = self.state.x[1]
        #vx = self.state.v[0]
        #vy = self.state.v[1]
        vy = action[0]
        vx = action[1]
        new_y = y + vy
        new_x = x + vx

        for i in range(1, 10):
            py = int(y + 0.5 + vy * i / 9)
            px = int(x + 0.5 + vx * i / 9)
            if self.out_of_track(py, px):
                return RT_State(True, self.track, [py, px], [vy, vx], self.dest), -100

        if(dist_point_line_passing_two_points([y, x], [new_y, new_x], self.dest) < 0.5):
            return RT_State(True, self.track, self.state.dest, [vy, vx], self.dest), 100

        self.state = RT_State(False, self.track, [new_y, new_x], [vy, vx], self.dest)
        return self.state, 0
