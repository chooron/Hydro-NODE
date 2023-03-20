# Hydro-NODE

Torch version for Hydro-NODE [https://github.com/marv-in/HydroNODE]

## Environment requirement

- python=3.7
- torch=1.13.1
- torchdiffeq=0.2.3
- pytorch-lightning=1.8.6

## How to run the experiment

- step 0, run `M0_optimization` to get the best parameters for the model
- step 1, run `M0_main.py` to get the train data, save in the `data`
- step 2, run `M50_main.py` for pretraining and training the M50 model
- step 3, run `M100_main.py` for pretraining and training the M100 model

## Note

- `utils` provides some training methods, such as `train`, `forecast` and base learner object `BaseLearner`

## My own idea

**A more neural training method for M50 and M100**

- Batch Training

    - How does the batch size affect the accuracy ?

        - incorrect nse evaluate for each batch training (is in consistent with the total series nse)

    - How to determine the S0 for each batch

        - constant S0

        - optimize S0

        - warm up method

        - Neural ODE

- Neural hydrology parameter

    - hyperparameter  (hyperparameter searching)

    - inner parameter (gradient optimization)

- impact factor

    - interpolate methods