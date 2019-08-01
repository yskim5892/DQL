import numpy as np
import utils
from DQNetwork import DQNetwork
import sys
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
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

class DQLearner:
    def __init__(self, hwc, ex_dim, action_dim, args, log_dir, save_dir, board_dir):
        self.net = DQNetwork(hwc, ex_dim, action_dim, args)
        
        self.net.build()
        self.net.build_train_op()
        try:
            self.net.restore(save_dir=save_dir)
        except(AttributeError, TypeError):
            self.net.initialize()

        self.action_dim = action_dim
        self.hwc = hwc
        self.h, self.w, self.c = hwc[0], hwc[1], hwc[2]
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.board_dir = board_dir
        self.args = args

    def epsilon_greedy_policy(self, state):
        if(np.random.random() <= self.args.epsilon):
            return None, np.random.randint(self.action_dim)
        else:
            Q = self.net.Q_value(state)
            return Q, np.argmax(Q)

    # task_specific
    def process_state(self, state):
        pass
    def decode_action(self, action):
        pass
    def is_success(self, state, reward):
        pass 

    def process_action(self, action):
        result = np.zeros([self.action_dim])
        result[action] = 1
        return result

    def learn(self, env):
        self.env = env
        self.writer = utils.SummaryWriter(self.board_dir)
        ep = 0
        sum_sum_reward, sum_avg_loss, sum_ep_length, max_sum_reward, max_avg_sum_reward = 0, 0, 0, float('-inf'), float('-inf')

        failure_record_history = []
        success_record_history = []
        
        while(True):
            env.initialize_environment()
            ep += 1
            sum_reward, sum_loss, n_loss, ep_length = 0, 0, 0, 0

            trajectory = []
            while(True):
                ep_length += 1
                state = self.process_state(env.state) 

                Q, action = self.epsilon_greedy_policy(state)
                next_state, reward= env.respond(self.decode_action(action))
                is_success = self.is_success(next_state, reward)
                is_terminal = next_state.is_terminal
                next_state = self.process_state(next_state)

                trajectory.append([state, self.decode_action(action), Q])
                record = Record(state, self.process_action(action), reward, next_state)
                if(is_success):
                    utils.queue_smart_put(success_record_history, record, self.args.max_experience)
                else:
                    utils.queue_smart_put(failure_record_history, record, self.args.max_experience)
                sum_reward += reward

                loss = self.net.learn_from_history(failure_record_history, success_record_history)
                if(loss != None):
                    sum_loss += loss
                    n_loss += 1

                if(is_terminal):
                    trajectory.append([next_state, None, None])
                    break
            
            if n_loss != 0:
                sum_avg_loss += sum_loss / n_loss
            sum_sum_reward += sum_reward
            sum_ep_length += ep_length
            if ep % self.args.print_ep_period == 0:
                avg_sum_reward = sum_sum_reward / self.args.print_ep_period
                avg_avg_loss = sum_avg_loss / self.args.print_ep_period
                avg_ep_length = sum_ep_length / self.args.print_ep_period

                print('episode : ', ep, ', avg_rew :', avg_sum_reward, ', avg_loss :', avg_avg_loss, ', avg_ep_length :', avg_ep_length)

                summary_dict = {"learning rate" : self.net.sess.run(self.net.lr), "average_reward" : avg_sum_reward, "average_loss" : avg_avg_loss, "average_episode_length" : avg_ep_length}
                self.writer.add_summaries(summary_dict, ep)

                if max_avg_sum_reward < avg_sum_reward:
                    self.net.save(ep, self.save_dir)
                    max_avg_sum_reward = avg_sum_reward

                sum_sum_reward, sum_avg_loss, sum_ep_length = 0, 0, 0
                self.save_trajectory(ep, trajectory)

            if max_sum_reward < sum_reward:
                max_sum_reward = sum_reward
                print('Reward ', sum_reward, ' from episode ', ep, '!')
                self.save_trajectory(ep, trajectory)                

    def save_trajectory(self, ep, trajectory):
        fname = self.log_dir + 'play' + str(ep) + '.pkl'
        utils.write_pkl(trajectory, fname)

