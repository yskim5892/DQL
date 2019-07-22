import numpy as np
#from DQNetwork import DQNetwork
from BHB_display import BHB_display

def max_kv_in_dict(d):
    max_key = None
    max_value = float('-inf')
    for key in d:
        if d[key] > max_value:
            max_key = key
            max_value = d[key]
    return max_key, max_value

class DQLearner:
    def __init__(self, state_shape, action_dim, args):
        #self.net = DQNetwork(state_shape, action_dim, args)
        #self.net.build()
        #self.net.build_train_op()
        self.action_dim = action_dim


    def epsilon_greedy_policy(self, state, epsilon=0.05):
        return np.random.randint(self.action_dim)


        '''Q = self.net.Q_value(state)
        r = np.random.random()

        if(r <= epsilon):
            return np.random.randint(self.action_dim)
        else:
            return np.argmax(Q)'''

    def learn(self, env):
        step = 0
        sum_sum_reward = 0

        state_history = []
        action_history = []
        reward_history = []
        next_state_history = []


        while(True):
            env.initialize_environment()
            step += 1
            sum_reward = 0

            trajectory = []

            while(True):
                state = env.state
                action = self.epsilon_greedy_policy(state)
                next_state, reward= env.respond(action)

                trajectory.append((state, action))
                state_history.append(state)
                action_history.append(action)
                reward_history.append(reward)
                next_state_history.append(next_state)

                sum_reward += reward

                #self.net.learn_from_history(state_history, action_history, reward_history, next_state_history)

                if(next_state.is_terminal):
                    break

            sum_sum_reward += sum_reward
            if step % 1 == 0:
                print('step : ', step, ', avg_rew :', sum_sum_reward / 1)
                sum_sum_reward = 0

                BHB_display(step, env.size, trajectory)
