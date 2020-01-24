import numpy as np
import math

def getMinLoss(self):
    min_loss_value = math.inf
    min_loss_key = (0,0,0)
    for key in self:
        loss_value = 128 - np.max(self[key])
        if loss_value < min_loss_value:
            min_loss_value = loss_value
            min_loss_key = key
    return(min_loss_key, min_loss_value)