
from typing import Tuple, Optional, Sequence, Union

import copy
import multiprocessing
import sys

from torchmetrics import Metric
import numpy as np
import torch


class RootMeanSquaredError(Metric):
    # Each state variable should be called using self.add_state(...)
    def __init__(self, dist_sync_on_step: bool = False, process_group = None, dist_sync_fn = None):
        super().__init__(
            #compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")  
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    
    @torch.jit.unused
    def forward(self, *args, **kwargs):
        """
        Automatically calls ``update()``. Returns the metric value over inputs if ``compute_on_step`` is True.
        """
        # add current step
        with torch.no_grad():
            self.update(*args, **kwargs)  # accumulate the metrics
        self._forward_cache = None

        """
        if self.compute_on_step:
            kwargs["just_return"] = True
            out_cache = self.update(*args, **kwargs)  # compute and return the rmse
            kwargs.pop("just_return", None)
            return out_cache
        """
        
    def update(self, preds, targs, just_return = False):  
        '''Any code needed to update the state given any inputs to the metric.'''

        # Reformat: targets should only contain NDVI!! Provide batch["target"][0][:,:,ndvi_idx,...]
        targs = targs.squeeze(3) # b t c
        
        # MSE computation    
        sum_squared_error = (((targs - preds))**2).sum((1,2))  # b c
        n_obs = preds.size(1) #Count number t in each batch
            
        if just_return:
            cubenames = targs["cubename"]
            rmse = torch.sqrt(sum_squared_error / n_obs)
            return [{"name":  cubenames[i], "rmse": rmse[i]} for i in range(len(cubenames))]
        else:
            self.sum_squared_error += sum_squared_error.sum()
            self.total += n_obs

    def compute(self):  
        """
        Computes a final value from the state of the metric.
        Computes mean squared error over state.
        """
        return {"RMSE_drought": torch.sqrt(self.sum_squared_error / self.total)}


class RMSE_drought(RootMeanSquaredError):

    def __init__(self, dist_sync_on_step: bool = False, process_group = None, dist_sync_fn = None):
        super().__init__(
            #compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn
        )
