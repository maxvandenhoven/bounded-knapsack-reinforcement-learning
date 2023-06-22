##########################################################################################
# Imports
##########################################################################################
import itertools
import warnings
warnings.filterwarnings("ignore")

from stable_baselines3 import PPO, A2C
from torch import nn

from src.training import train_model


##########################################################################################
# Grid search
##########################################################################################
GRID = {
    "algorithm_class": [PPO, A2C],
    "gamma": [0.99, 0.9],
    "learning_rate": [0.0003, 0.001],
    "normalize_env": [True, False],
    "activation_fn": [nn.LeakyReLU, nn.ReLU],
    "net_arch": [
        [256, 128],
        [256, 256],
        [512, 256],
        [512, 256, 128],
    ],
}

def hyperparam_generator(grid: dict[str, list]):
    keys = list(grid.keys())
    values = list(grid.values())
    combinations = list(itertools.product(*values))

    for combination in combinations:
        yield dict(zip(keys, combination))


##########################################################################################
# Training loop
##########################################################################################
def train_models():
    for hyperparams in hyperparam_generator(GRID):
        train_model(
            total_timesteps=100000,
            verbose=0,
            **hyperparams
        )


##########################################################################################
# Main
##########################################################################################
if __name__ == "__main__":
    train_models()
