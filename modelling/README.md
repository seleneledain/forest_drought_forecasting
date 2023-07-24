# Drought Impact Forecasting 

A PyTorch lightning library for Earth surface forecasting.

This library contains models, dataloaders and scripts for Earth surface forecasting (specifically NDVI for drought impacts). This code is based off the work developed in the [EarthNet](www.earthnet.tech) challenge and follows a similar approach and structure.

It is currently under development, thus do expect bugs and please report them!

The library is build on [PyTorch](www.pytorch.org), a Python deep learning library, and [PyTorch Lightning](https://www.pytorchlightning.ai/), a PyTorch wrapper reducing boilerplate code and adding functionality to scale experiments.

There is three main components:

    1. Model - plain PyTorch models just implementing simple forward passes.
    2. Setting - Dataset and Metrics for a particular problem
    3. Task - Abstraction for the training, validation & test loops, tying together models and settings, normally both models and settings are task-specific (i.e. the Pytorch Lightning module)


## Requirements

We recommend using Anaconda for managing dependencies of this library. The following bash commands create a suitable environment. Please note the PyTorch installation requirements for your system, see (https://pytorch.org/) - esp. cudatoolkit might have to be installed with a different cuda version.

```
conda env create -f environment.yml
conda activate drought
```

## Repository structure

```.
├── README.md                               > This file contains a description of what is the repository used for and how to use it.
├── configs                                 > Stores the config files for the modelling/inference.
    ├── drought/drought_lstm/base.yaml      > Structure to store config file configs/<setting>/<model>/config.yaml. This one is the training file.
    ├── drought/drought_lstm/params.yaml    > Values to test in hyperparameter tuning.
├── earthnet_models_pytorch                 > Folder with models, metrics, pipelines...
    ├── model                               
        ├── drought_lstm.py                 > Multistep LSTM for foreacsting NDVI based on spatio temporal timeseries. 
    ├── setting 
        ├── drought_data.py                 > Dataset class and dataloader.
        ├── drought_metric.py               > Definition of validation metric specific to task (RMSE here).
    ├── task
        ├── loss.py                         > Definition of loss functions (MSE here).
        ├── ~~schedule.py~~                 > Schedulers to optimise learning rate during training.
        ├── spatio_temporal.py              > Pytorch Lightning module where training, validation and test steps are defined.
    ├── utils
        ├── parse.py                        > For reading yaml config files.
    ├── experiments                         > Stores model results and checkpoints.
    ├── scripts
        ├── train.py                        > Train a model.
        ├── tune.py                         > Perform hyperparameter grid search.
        ├── test.py                         > Inference on a model.
        ├── debug.py                        > Short training loop to check if model works.
```


## Train


In order to train a model we need to set up a `config.yaml`, see above regarding for more details.

1. If you are not using the context and target lengths of the data samples as setup in `base.yaml`, then you need change them also in `task.__init__.py`.
2. Your training data folder should be named `train`. If not, you must adapt the folder name in the `setting.drought_data.setup` function.
3. Then we just do:
```
python scripts/train.py configs/drought/drought_lstm/base.yaml
```

It trains the model as specified in the config. Validation is also done, by using a fraction of the test set (HOW TO CHANGE VAL SIZE).

To view results/training progress:
```
conda activate drought
cd /path_to/experiments/drought/drought-lstm/drought_lstm/
tensorboard --logdir your_exp
```



## Tune

Hyperparameter tuning can be performed by providing lists of values for each model parameter. These should be defined in `configs/params.yaml`. Each combination will be trained using a grid search. The best configuration will be written automatically to `configs/best.yaml` which is ready to use for training.

To tune we just do:
```
python scripts/tune.py configs/drought/drought_lstm/params.yaml configs/drought/drought_lstm/base.yaml
```

Each combination will have its own results saved to `experiments`. A comparison of the various trained model scores can be  found in `experiments/tune_results.csv`.


## Test 

To test a model you have trained. 

1. You might have different testing tracks/datasets. You can add their names as well as context and target lengths in `task.__init__.py`
2. Then we just do:
```
python scripts/test.py --setting configs/drought/drought_lstm/base.yaml --checkpoint experiments/drought/drought-lstm/drought_lstm/full_train/checkpoints/last.ckpt --track track_name --pred_dir 'path_to_store_predictions’
```

To view results/testing progress:
```
conda activate drought
cd /path_to/experiments/drought/drought-lstm/drought_lstm/
tensorboard --logdir your_exp
```


## Debug

The design process of a new model or feature is supported by a debug option. Since models often require both a lot of data and GPUs to test the complete training cycle, we use this debug option rather than classic unit testing.
In order to use it, we need to set up a `config.yaml` containing all configs for the different components. See the `configs` folder for examples. It is recommended to save the setting in a folder structure `configs/<setting>/<model>/base.yaml`. If done in this way, the logger automatically detects the correct naming for later.

We can check if a model works as desired by running:
```
python scripts/debug.py path/to/setting.yaml
```

It starts with a fast dev run in PyTorch lightning, which is essentially just performing two train, validation and test loops. It then also overfits a model on 4 batches for 1000 epochs, to check if gradients flow properly and the model does indeed learn. 

