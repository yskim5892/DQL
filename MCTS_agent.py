import numpy as np
import utils
from DQNetwork import DQNetwork
import sys
import Learners
#from BHB_display import BHB_display

def max_kv_in_dict(d):
    max_key = None
    max_value = float('-inf')
    for key in d:
        if d[key] > max_value:
            max_key = key
            max_value = d[key]
    return max_key, max_value

class Record:
    def __init__(self, state, action, G, later_state):
        self.state = state
        self.action = action
        self.G = G
        self.later_state = later_state

class MCTS_agent:
    def __init__(self, hwc, action_dim, args, log_dir):
        self.action_dim = action_dim
        self.hwc = hwc
        self.h, self.w, self.c = hwc[0], hwc[1], hwc[2]
        self.log_dir = log_dir
        self.args = args

    def policy(self):
        actions_candidate = list(range(self.w))
        high_score_actions = []
        for x in range(self.w):
            grade = self.env.grade_for_action(x)
            if(0 < grade and grade < 3):
                actions_candidate.remove(x)
                continue
            if(grade == 3):
                high_score_actions.append(x)

        if(len(high_score_actions) >= 1):
            return np.random.choice(high_score_actions)
        elif(len(actions_candidate) >= 1):
            return np.random.choice(actions_candidate)
        else:
            return np.random.randint(self.action_dim)
    
    def process_state(self, state):
        if(self.args.task == 'BHB'):
            return Learners.process_state_BHB(self.hwc, state)
        elif(self.args.task == 'RT'):
            return Learners.process_state_RT(self.hwc, self.env, state)

    def decode_action(self, action):
        if(self.args.task == 'BHB'):
            return Learners.decode_action_BHB(action)
        elif(self.args.task == 'RT'):
            return Learners.decode_action_RT(action)

    def run(self, env):
        self.env = env
        ep = 0
        sum_sum_reward, sum_ep_length, max_sum_reward, max_avg_sum_reward = 0, 0, float('-inf'), float('-inf')
        
        while(True):
            env.initialize_environment()
            ep += 1
            sum_reward, sum_loss, n_loss, t = 0, 0, 0, 0

            trajectory = [] # list of pair of (state, action, reward)
            while(True):
                t += 1
                state = self.process_state(env.state) 

                action = self.policy()
                next_state, reward= env.respond(self.decode_action(action))
                is_terminal = next_state.is_terminal
                next_state = self.process_state(next_state)
                trajectory.append([state, self.decode_action(action), reward])
                
                sum_reward += reward

                if(is_terminal):
                    trajectory.append([next_state, None, None])
                    break
            
            sum_sum_reward += sum_reward
            sum_ep_length += t
            if ep % self.args.print_ep_period == 0:
                avg_sum_reward = sum_sum_reward / self.args.print_ep_period
                avg_ep_length = sum_ep_length / self.args.print_ep_period

                print('episode : ', ep, ', avg_rew :', avg_sum_reward, ', avg_ep_length :', avg_ep_length)

                if max_avg_sum_reward < avg_sum_reward:
                    max_avg_sum_reward = avg_sum_reward

                sum_sum_reward, sum_ep_length = 0, 0
                self.save_trajectory(ep, trajectory)

            if max_sum_reward < sum_reward:
                max_sum_reward = sum_reward
                print('Reward ', sum_reward, ' from episode ', ep, '!')
                self.save_trajectory(ep, trajectory)                

    def save_trajectory(self, ep, trajectory):
        fname = self.log_dir + 'play' + str(ep) + '.pkl'
        utils.write_pkl(trajectory, fname)

