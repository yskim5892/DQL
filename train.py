from Bauhausbreak import BHB_Environment
from DQLearner import DQLearner
import utils

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--gpu", default = 0, help="Utilize which gpu", type = int)
parser.add_argument("--gamma", default = 0.95, type = float)
parser.add_argument("--lr", default = 0.1, help="learning rate", type=float)
parser.add_argument("--decay_steps", default = 100000, type=int)
parser.add_argument("--decay_rate", default = 0.9, type=float)
parser.add_argument("--max_experience", default = 1000, type=int)
parser.add_argument("--batch_size", default = 256, type=int)

args = parser.parse_args()

if __name__ == '__main__':
    name_format = utils.NameFormat('lr', 'gamma', 'batch_size')
    FILE_ID = name_format.get_id_from_args(args)
    RESULT_DIR = '/home/yskim5892/DQL_results/'
    LOG_DIR = RESULT_DIR + '/log/%s/'%FILE_ID
    SAVE_DIR = RESULT_DIR + '/model/%s/'%FILE_ID
    utils.create_muldir(RESULT_DIR, SAVE_DIR, LOG_DIR)

    size = 8
    BHB_env = BHB_Environment(size)
    agent = DQLearner([size, size, 31], 2, size, args, LOG_DIR, SAVE_DIR)
    agent.learn(BHB_env)
