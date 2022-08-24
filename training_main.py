import argparse

from train.general_training import train
from utils.config_parser import ConfigParser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to the config file to use for training")

    args = parser.parse_args()
    train(ConfigParser.read(args.config))
