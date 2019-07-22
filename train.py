from Bauhausbreak import BHB_Environment
from DQLearner import DQLearner

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--gpu", default = 0, help="Utilize which gpu", type = int)
parser.add_argument("--gamma", default = 1, type = float)
parser.add_argument("--lr", default = 1e-2, help="learning rate", type=float)
parser.add_argument("--decay_step", default = 1000, type=int)
parser.add_argument("--decay_rate", default = 0.7, type=float)
parser.add_argument("--max_experience", default = 1000, type=int)
parser.add_argument("--batch_size", default = 128, type=int)

args = parser.parse_args()

if __name__ == '__main__':
    BHB_env = BHB_Environment(1)
    agent = DQLearner([8, 8, 1], 8, args)
    agent.learn(BHB_env)