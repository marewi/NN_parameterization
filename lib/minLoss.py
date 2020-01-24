import numpy as np

def getMinLoss(self):
    min_loss_value = 0
    min_loss_key = (0,0,0)
    for key in self:
        loss_value = np.min(self[key])
        if loss_value > min_loss_value:
            min_loss_value = loss_value
            min_loss_key = key
    return(min_loss_key, min_loss_value)