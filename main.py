from train.general_training import train
from utils.config_parser import ConfigParser


if __name__ == '__main__':
    # train(ConfigParser.read("../configs/debugging_autoencoder.yaml"))
    train(ConfigParser.read("./configs/debugging_diffusion_model.yaml"))
