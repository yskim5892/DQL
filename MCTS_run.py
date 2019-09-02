from Bauhausbreak import BHB_Environment
from Racetrack import RT_Environment
from MCTS_agent import MCTS_agent
import utils

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--task", default = 'BHB', type=str)
parser.add_argument("--size", default = 8, help="environment size", type=int)

args = parser.parse_args()

if __name__ == '__main__':
    name_format = utils.NameFormat('size')
    FILE_ID = name_format.get_id_from_args(args)
    RESULT_DIR = '/home/yskim5892/DQL_results/%s/'%args.task
    LOG_DIR = RESULT_DIR + '/log/%s/'%FILE_ID
    utils.create_muldir(LOG_DIR)

    if args.task == 'BHB':
        args.print_ep_period = 50
        BHB_env = BHB_Environment(args.size)
        agent = MCTS_agent([args.size, args.size, 16], args.size, args, LOG_DIR)
        agent.run(BHB_env)

