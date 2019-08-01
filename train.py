from Bauhausbreak import BHB_Environment
from Racetrack import RT_Environment
from Learners import BHBLearner, RTLearner
import utils

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--task", type=str)
parser.add_argument("--gpu", default = 0, help="Utilize which gpu", type = int)
parser.add_argument("--gamma", default = 0.95, type = float)
parser.add_argument("--epsilon", default = 0.1, type = float)
parser.add_argument("--lr", default = 0.001, help="learning rate", type=float)
parser.add_argument("--target_update_period", default = 2000, type=int)
parser.add_argument("--decay_steps", default = 1e7, type=int)
parser.add_argument("--decay_rate", default = 0.9, type=float)
parser.add_argument("--max_experience", default = 20000, type=int)
parser.add_argument("--batch_size", default = 256, type=int)
parser.add_argument("--size", default = 8, help="environment size", type=int)

args = parser.parse_args()

if __name__ == '__main__':
    name_format = utils.NameFormat('lr', 'epsilon', 'gamma', 'batch_size', 'target_update_period', 'size')
    FILE_ID = name_format.get_id_from_args(args)
    RESULT_DIR = '/home/yskim5892/DQL_results/%s/'%args.task
    LOG_DIR = RESULT_DIR + '/log/%s/'%FILE_ID
    SAVE_DIR = RESULT_DIR + '/model/%s/'%FILE_ID
    BOARD_DIR = RESULT_DIR + '/board/%s/'%FILE_ID
    utils.create_muldir(RESULT_DIR, SAVE_DIR, LOG_DIR)

    if args.task == 'BHB':
        args.print_ep_period = 50
        BHB_env = BHB_Environment(args.size)
        agent = BHBLearner([args.size, args.size, 16], 11, args.size, args, LOG_DIR, SAVE_DIR, BOARD_DIR)
        agent.learn(BHB_env)

    elif args.task == 'RT':
        args.print_ep_period = 1000
        RT_env = RT_Environment(args.size)
        agent = RTLearner([args.size, args.size, 4], 7, 9, args, LOG_DIR, SAVE_DIR, BOARD_DIR)
        agent.learn(RT_env)
