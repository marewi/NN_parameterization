from torch import nn

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# loss function

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

        self.shape_c = nn.CrossEntropyLoss()
        self.color_c = nn.CrossEntropyLoss()
        self.coord_c = nn.SmoothL1Loss()

    def forward(self, pred, target):
        pshape, pcolor, pcoord = pred
        tshape, tcolor, tcoord = target

        shape_loss = self.shape_c(pshape, tshape)
        color_loss = self.color_c(pcolor, tcolor)
        coord_loss = self.coord_c(pcoord, tcoord)

        loss = shape_loss + color_loss + coord_loss
        loss_dict = {'shape': shape_loss, 'color': color_loss, 'coord': coord_loss}
        return loss, loss_dict
