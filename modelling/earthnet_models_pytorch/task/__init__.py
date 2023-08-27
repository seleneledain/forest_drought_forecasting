


from earthnet_models_pytorch.task.loss import setup_loss
from earthnet_models_pytorch.task.shedule import SHEDULERS
from earthnet_models_pytorch.task.spatio_temporal import SpatioTemporalTask

TASKS = {
    "spatio-temporal": SpatioTemporalTask
}


TRACK_INFO = {
    "drought": {
        "iid": {
            "context_length": 54, # (9 months)
            "target_length": 18, #(3 months)
        },
        "ood": { #out of domain
            "context_length": 54, # (9 months)
            "target_length": 18, #(3 months)
        }
    }
}
