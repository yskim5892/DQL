from Racetrack import RT_Environment
from DQLearner import DQLearner
import utils
import numpy as np
import pickle

class dummy(object):
    pass

def get_default_args():
    args = dummy()
    args.task = 'RT'
    args.gpu = 0
    args.gamma = 0.95
    args.epsilon = 0.05
    args.lr = 0.0001
    args.target_update_period = 2000
    args.decay_steps = 1e7
    args.decay_rate = 0.9
    args.max_experience = 20000
    args.batch_size = 256
    args.size = 16
    args.n = 3
    return args

def get_env_agent():
    args = get_default_args()
    name_format = utils.NameFormat('lr', 'epsilon', 'gamma', 'n', 'batch_size', 'target_update_period', 'size')
    FILE_ID = name_format.get_id_from_args(args)
    RESULT_DIR = '/home/yskim5892/DQL_results/%s/'%args.task
    LOG_DIR = RESULT_DIR + '/log/%s/'%FILE_ID
    SAVE_DIR = RESULT_DIR + '/model/%s/'%FILE_ID
    BOARD_DIR = RESULT_DIR + '/board/%s/'%FILE_ID

    size = args.size
    with open(LOG_DIR + 'play1000.pkl', 'rb') as f:
        trajectory = np.array(pickle.load(f))
        track = np.reshape(trajectory[0][0][0:size*size*3], [size, size, 3])
        track = track[:, :, 0]
        '''track = np.zeros([size, size], dtype=np.dtype(int))
        for i in range(size):
            for j in range(size):
                if(track_[i][j][0] == 1):
                    track[i][j] = 1'''

    env = RT_Environment(16, track)
    agent = DQLearner([16, 16, 3], 5, 9, args, LOG_DIR, SAVE_DIR, BOARD_DIR)
    agent.env = env
    return env, agent

