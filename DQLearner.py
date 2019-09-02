import numpy as np
import utils
from Logger import Logger
from DQNetwork import DQNetwork
import Learners
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
    def __init__(self, state, action, G, later_state):
        self.state = state
        self.action = action
        self.G = G
        self.later_state = later_state

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
        self.ex_dim = ex_dim
        self.img_dim = self.h * self.w * self.c
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.logger = Logger(log_dir + '/train_log')
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
        if(self.args.task == 'BHB'):
            return Learners.process_state_BHB(self.hwc, state)
        elif(self.args.task == 'RT'):
            return Learners.process_state_RT(self.hwc, self.env, state)

    def decode_action(self, action):
        if(self.args.task == 'BHB'):
            return Learners.decode_action_BHB(action)
        elif(self.args.task == 'RT'):
            return Learners.decode_action_RT(action)

    def is_success(self, state, reward):
        if(self.args.task == 'BHB'):
            return Learners.is_success_BHB(state, reward)
        elif(self.args.task == 'RT'):
            return Learners.is_success_RT(state, reward)

    def process_action(self, action):
        result = np.zeros([self.action_dim])
        result[action] = 1
        return result

    def record_n_steps_record(self, history, trajectory, next_state, t, n):
        tau = t - n
        G = 0
        for i in range(tau, tau + n):
            G += pow(self.args.gamma, i - tau) * trajectory[i][2]
        state_n_steps_ago = trajectory[tau][0]
        action_n_steps_ago = trajectory[tau][1]
        record = Record(state_n_steps_ago, self.process_action(action_n_steps_ago), G, next_state)
        utils.queue_smart_put(history, record, self.args.max_experience)

    def learn(self, env):
        self.env = env
        self.writer = utils.SummaryWriter(self.board_dir)
        ep = 0
        n = self.args.n
        sum_sum_reward, sum_avg_loss, sum_ep_length, max_sum_reward, max_avg_sum_reward = 0, 0, 0, float('-inf'), float('-inf')

        failure_record_history = []

        try:
            success_record_history = self.env.make_success_records(self.args.batch_size, self.args.n, self.args.gamma, self.process_state, self.decode_action, self.process_action)
        except AttributeError:
            print('Failed to call make_succes_records funtion!')
            success_record_history = []
        temp_i = 0
        while(True):
            env.initialize_environment()
            ep += 1
            sum_reward, sum_loss, n_loss, t = 0, 0, 0, 0

            trajectory = [] # list of pair of (state, action, reward, Q)
            while(True):
                t += 1
                state = self.process_state(env.state) 

                Q, action = self.epsilon_greedy_policy(state)
                next_state, reward= env.respond(self.decode_action(action))
                is_success = self.is_success(next_state, reward)
                is_terminal = next_state.is_terminal
                next_state = self.process_state(next_state)

                trajectory.append([state, action, reward, Q])
                
                if(is_success):
                    history = success_record_history
                else:
                    history = failure_record_history

                if(is_terminal):
                    for i in range(1, min(n+1, t)):
                        self.record_n_steps_record(history, trajectory, next_state, t, i)
                elif t >= n:
                    self.record_n_steps_record(history, trajectory, next_state, t, n)
                
                sum_reward += reward

                loss = self.net.learn_from_history(failure_record_history, success_record_history, self.logger)
                if(loss != None):
                    sum_loss += loss
                    n_loss += 1

                if(is_terminal):
                    trajectory.append([next_state, None, None])
                    break
            
            if n_loss != 0:
                sum_avg_loss += sum_loss / n_loss
            sum_sum_reward += sum_reward
            sum_ep_length += t
            if ep % self.args.print_ep_period == 0:
                avg_sum_reward = sum_sum_reward / self.args.print_ep_period
                avg_avg_loss = sum_avg_loss / self.args.print_ep_period
                avg_ep_length = sum_ep_length / self.args.print_ep_period

                self.logger.log('episode : ' + str(ep) + ', avg_rew :' + str(avg_sum_reward) + ', avg_loss :' + str(avg_avg_loss) + ', avg_ep_length :' + str(avg_ep_length))

                summary_dict = {"learning rate" : self.net.sess.run(self.net.lr), "average_reward" : avg_sum_reward, "average_loss" : avg_avg_loss, "average_episode_length" : avg_ep_length}
                self.writer.add_summaries(summary_dict, ep)

                if max_avg_sum_reward < avg_sum_reward:
                    self.net.save(ep, self.save_dir)
                    max_avg_sum_reward = avg_sum_reward

                sum_sum_reward, sum_avg_loss, sum_ep_length = 0, 0, 0
                self.save_trajectory(ep, trajectory)

            if max_sum_reward < sum_reward:
                max_sum_reward = sum_reward
                self.logger.log('Reward ' + str(sum_reward) + ' from episode ' + str(ep) + '!')
                self.save_trajectory(ep, trajectory)                

    def save_trajectory(self, ep, trajectory):
        fname = self.log_dir + 'play' + str(ep) + '.pkl'
        utils.write_pkl(trajectory, fname)

