# Bounded Knapsack with Reinforcement Learning

Note: all commands below should be run from the repositories' top-level directory.

To make a virtual environment with conda, use the following command:
```
conda env create --name rl_env --file environment.yml
```

To view the logs in tensorboard, use the following command:
```
tensorboard --logdir logs
```

To train the models using grid search, use the following command:
```
python train.py
```

To test a specific model from the `logs` directory, use the following command:
```
python test.py
```
and specify the model's corresponding folder name when prompted. Please note that we use a `EvalCallback` to save models when they improve the mean episode reward. This appears to lead to some instability when loading the models however, as testing some models yields a very low reward around 200, indicating that the weights have not been properly loaded. Unfortunately, we did not figure out a way to overcome this, so we kindly ask you to trust the tensorboard evaluation metrics in case the testing script returns very poor results.