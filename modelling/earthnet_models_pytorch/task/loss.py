

import sys

from typing import Optional, Union

import torch

from torch import nn
import torch.nn.functional as F
import torch.distributions as distrib

from earthnet_models_pytorch.task.shedule import WeightShedule


def make_normal_from_raw_params(raw_params, scale_stddev=1, dim=-1, eps=1e-8):
    """
    Creates a normal distribution from the given parameters.

    Parameters
    ----------
    raw_params : torch.Tensor
        Tensor containing the Gaussian mean and a raw scale parameter.
    scale_stddev : float
        Multiplier of the final scale parameter of the Gaussian.
    dim : int
        Dimensions of raw_params so that the first half corresponds to the mean, and the second half to the scale.
    eps : float
        Minimum possible value of the final scale parameter.

    Returns
    -------
    torch.distributions.Normal
        Normal distributions with the input mean and eps + softplus(raw scale) * scale_stddev as scale.
    """
    loc, raw_scale = torch.chunk(raw_params, 2, dim)
    assert loc.shape[dim] == raw_scale.shape[dim]
    scale = F.softplus(raw_scale) + eps
    normal = distrib.Normal(loc, scale * scale_stddev)
    return normal



class L2NDVILoss(nn.Module):

    def __init__(self, ndvi_pred_idx = 0, ndvi_targ_idx = 0, **kwargs):
        super().__init__()
        
        self.ndvi_pred_idx = ndvi_pred_idx
        self.ndvi_targ_idx = ndvi_targ_idx

    def forward(self, preds, batch, current_step = None):

        
        ndvi_targ = batch["target"][0][:, :, self.ndvi_targ_idx,...].squeeze(3) # b t c
        ndvi_pred = preds[:,:,self.ndvi_pred_idx, ...].unsqueeze(2) # b t c 
        sum_squared_error = (((ndvi_targ - ndvi_pred))**2).sum(1)  # b c
        mse = sum_squared_error / ndvi_pred.size(1) # b c 
        mse_mean = mse.mean()
        logs = {"loss": mse_mean}

        return mse_mean, logs


LOSSES = {"l2": L2NDVILoss}


def setup_loss(args):

    if args["name"] == "L2NDVILoss":
        return L2NDVILoss(**args)
    
    else:  
        raise Exception("Not a valid loss function.")