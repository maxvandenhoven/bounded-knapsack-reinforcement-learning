##########################################################################################
# Imports
##########################################################################################
import os
import textwrap
import warnings
warnings.filterwarnings("ignore")

import or_gym
import numpy as np
from or_gym.envs.classic_or.knapsack import BoundedKnapsackEnv

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy

from src.training import MAX_WEIGHT, NormalizingWrapper


##########################################################################################
# Configuration and reprodubility
##########################################################################################
# Set seed for reproducibility
np.random.seed(42)


##########################################################################################
# Testing function
##########################################################################################
def test_model():
    # Ask user what model to evaluate
    model_log_folder = input(textwrap.dedent(
        """
        To evaluate a model, type the name of the 
        corresponding folder under the logs/ directory): """))
    print()
    
    # Check if specified model exist in the logs/ directory
    if model_log_folder not in os.listdir("logs/"):
        raise Exception("Specified model folder does not exist under logs/")
    
    # Get hyperparameters needed for evaluation from folder name
    algorithm_class_name, _, _, normalize_env, _, _ = model_log_folder.split("-")
    normalize_env = bool(normalize_env)
    if algorithm_class_name == "PPO":
        algorithm_class = PPO
    elif algorithm_class_name == "A2C":
        algorithm_class = A2C
    else:
        raise Exception("Unknown algorithm class name")

    # Load model
    model = algorithm_class.load(f"logs/{model_log_folder}/best_model")

    # Make environment and apply normalization wrapper if specified
    env: BoundedKnapsackEnv = or_gym.make(
        "Knapsack-v2", max_weight=MAX_WEIGHT, mask=False
    )
    if normalize_env: env = NormalizingWrapper(env)

    # Evaluate model
    mean_reward, std_reward = evaluate_policy(
        model=model, 
        env=env, 
        n_eval_episodes=100, 
        deterministic=False
    )
    print(f"Mean reward over 100 episodes: {mean_reward:.2f} +/- {std_reward:.2f}")


##########################################################################################
# Main
##########################################################################################
if __name__ == "__main__":
    test_model()
