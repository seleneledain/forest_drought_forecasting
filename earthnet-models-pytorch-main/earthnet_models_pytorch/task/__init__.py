


from earthnet_models_pytorch.task.loss import setup_loss
from earthnet_models_pytorch.task.shedule import SHEDULERS
from earthnet_models_pytorch.task.spatio_temporal import SpatioTemporalTask

TASKS = {
    "spatio-temporal": SpatioTemporalTask
}


TRACK_INFO = {
    "drought": {
        "iid": {
            "context_length": 3, #55, # 9 months at 5-daily
            "target_length": 2 #20 # 3 months at 5-daily
        },
        "ood": {
            "context_length": 55, # 9 months at 5-daily
            "target_length": 20 # 3 months at 5-daily
        }
    }
}
