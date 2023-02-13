# Hydro-NODE
Torch version for Hydro-NODE [https://github.com/marv-in/HydroNODE]




## Environment requirement

- python=3.7
- torch=1.13.1
- torchdiffeq=0.2.3
- pytorch-lightning=1.8.6

## How to run the experiment

- step 1, run `run_exp_hydro.py` to get the train data, save in the `data`
- step 2, run `M50_main.py` for pretraining and training the M50 model
- step 3, run `M100_main.py` for pretraining and training the M100 model

## Note

- `utils` provides some training methods, such as `train`, `forecast` and base learner object `BaseLearner`

- This project only implements the training process of the Hydro-NODE, the testing is to be continued

- The accuracy of the M50 and M100 don't reach the origin project, some modify (GPU support) will also come in soon
  
  
