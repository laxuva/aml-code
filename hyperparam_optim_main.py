import argparse

from hyperparam_opt.optimize_dm_training import optimize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="path to the config file to use for training"
    )
    args = parser.parse_args()

    optimize(args.config, "results")
