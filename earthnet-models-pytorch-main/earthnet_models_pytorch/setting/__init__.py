
from earthnet_models_pytorch.setting.drought_data import DroughtDataModule

from earthnet_models_pytorch.setting.drought_metric import RMSE_drought


SETTINGS = ["drought"]
METRICS = {"drought":RMSE_drought}
DATASETS = {"drought":DroughtDataModule}
METRIC_CHECKPOINT_INFO ={
    "drought": {
        "monitor": "RMSE_drought",
        "filename": 'Epoch-{epoch:02d}-RMSE-{RMSE:.4f}',
        "mode": 'min'
    }
}
