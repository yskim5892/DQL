import numpy as np
import utils
from DQNetwork import DQNetwork
#from BHB_display import BHB_display

def max_kv_in_dict(d):
    max_key = None
    max_value = float('-inf')
    for key in d:
        if d[key] > max_value:
            max_key = key
            max_value = d[key]
    return max_key, max_value

class DQLearner:
    def __init__(self, hwc, ex_dim, action_dim, args, log_dir, save_dir):
        self.net = DQNetwork(hwc, ex_dim, action_dim, args)
        self.net.build()
        self.net.build_train_op()
        self.action_dim = action_dim
        self.hwc = hwc
        self.h, self.w, self.c = hwc[0], hwc[1], hwc[2]
        self.save_dir = save_dir
        self.log_dir = log_dir

    def epsilon_greedy_policy(self, state, epsilon=0.05):
        Q = self.net.Q_value(state)
        r = np.random.random()

        if(r <= epsilon):
            return np.random.randint(self.action_dim)
        else:
            return np.argmax(Q)

    def process_state(self, state):
        blocks = np.zeros(self.hwc)
        for y in range(self.h):
            for x in range(self.w):
                blocks[y][x][state.blocks[x][y]] = 1

        blocks = np.reshape(blocks, [self.h * self.w * self.c])

        return np.concatenate((blocks, [state.gauge], [state.current_block]), 0)

    def process_action(self, action):
        result = np.zeros([self.w])
        result[action] = 1
        return result

    def learn(self, env):
        ep = 0
        print_ep_period = 10
        sum_sum_reward = 0
        sum_avg_loss = 0
        max_sum_reward = -1

        state_history = []
        action_history = []
        reward_history = []
        next_state_history = []
        
        self.net.initialize()
        while(True):
            env.initialize_environment()
            ep += 1
            sum_reward = 0
            sum_loss = 0
            n_loss = 0

            trajectory = []
            while(True):
                state = self.process_state(env.state) 

                action = self.epsilon_greedy_policy(state)
                next_state, reward= env.respond(action)
                is_terminal = next_state.is_terminal
                next_state = self.process_state(next_state)

                trajectory.append([state, action])
                state_history.append(state)
                action_history.append(self.process_action(action))
                reward_history.append(reward)
                next_state_history.append(next_state)

                sum_reward += reward

                loss = self.net.learn_from_history(state_history, action_history, reward_history, next_state_history)
                if(loss != None):
                    sum_loss += loss
                    n_loss += 1

                if(is_terminal):
                    break
            
            if n_loss != 0:
                sum_avg_loss += sum_loss / n_loss
            sum_sum_reward += sum_reward
            if ep % print_ep_period == 0:
                print('episode : ', ep, ', avg_rew :', sum_sum_reward / print_ep_period, ', avg_loss :', sum_avg_loss / print_ep_period)
                sum_sum_reward = 0
                sum_avg_loss = 0

            if max_sum_reward < sum_reward:
                max_sum_reward = sum_reward
                print('Reward ', sum_reward, ' from episode ', ep, '!')
                self.net.save(ep, self.save_dir)
                
                fname = self.log_dir + 'play' + str(ep)
                f = open(fname, 'w')
                f.write(str(trajectory))
                f.close()

