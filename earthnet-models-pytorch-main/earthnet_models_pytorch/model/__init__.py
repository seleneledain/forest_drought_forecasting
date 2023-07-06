
from earthnet_models_pytorch.model.drought_lstm import LSTM_oneshot
from earthnet_models_pytorch.task import TASKS

MODELS = {
    "drought-lstm": LSTM_oneshot
}

MODELTASKS = {
    "drought-lstm": TASKS["spatio-temporal"]
}

MODELTASKNAMES = {
    "drought-lstm": "spatio-temporal"
}