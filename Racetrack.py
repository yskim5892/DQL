from Environment import *
from utils import *
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
    def __init__(self, size):
        self.size = size

        self.construct_track()
        self.initialize_environment()
    
    def construct_track(self):
        n = self.size
        track = np.zeros([n, n], dtype=np.dtype(int)) # 0 : empty, 1 : track

        prev_x_min = 0
        prev_x_max = n - 1
        self.track_poses = []
        for y in range(0, n):
            while(True):
                x = np.random.randint(0, n - 2)
                w = np.random.randint(1, n - x + 1)
                if(not (x > prev_x_max or x + w - 1 < prev_x_min)):
                    break
            prev_x_min = x
            prev_x_max = x + w - 1
            for i in range(x, x+w):
                track[i][y] = 1
                self.track_poses.append([i, y])
        
        ind = np.random.choice(range(len(self.track_poses)))
        self.dest = self.track_poses[ind]

        self.track = track

    def initialize_environment(self):
        ind = np.random.choice(range(len(self.track_poses)))
        x = self.track_poses[ind]
        self.state = RT_State(False, self.track, x, [0, 0], self.dest)

    def out_of_track(self, x, y):
        return x < 0 or x >= self.size or y < 0 or y >= self.size or self.track[x][y] == 0

    def respond(self, action):
        x = self.state.x[0]
        y = self.state.x[1]
        vx = self.state.v[0]
        vy = self.state.v[1]
        vx += action[0]
        vy += action[1]
        new_x = x + vx
        new_y = y + vy

        for i in range(1, 11):
            px = int(round(x + vx * i / 10))
            py = int(round(y + vy * i / 10))
            if self.out_of_track(px, py):
                return RT_State(True, self.track, [px, py], [vx, vy], self.state.dest), -100

        if(dist_point_line_passing_two_points([x, y], [new_x, new_y], self.state.dest) < 0.5):
            return RT_State(True, self.track, self.state.dest, [vx, vy], self.state.dest), 100

        self.state = RT_State(False, self.track, [new_x, new_y], [vx, vy], self.state.dest)
        return self.state, -1
