import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from random import shuffle

shape2label = {'triangle':0, 'windows':1, 'hook':2}
color2label = {'red':0, 'green':1, 'blue':2}
label2shape = {0:'triangle', 1:'windows', 2:'hook'}
label2color = {0:'red', 1:'green', 2:'blue'}


# saving sample outputs (images)
def imshow(img, idx):
    os.makedirs('sample_outputs', exist_ok=True)
    img = img / 2 + 0.5
    npimg = img.cpu().numpy()
    img = np.transpose(npimg, (1, 2, 0))
    plt.imshow(img)
    plt.savefig(f'sample_outputs/sample_output_{idx:06d}.png')
    # plt.show()

class Data(Dataset):
    def __init__(self,
                 is_train,
                 transform = transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                 ])):

        fname = 'neural_network/train_result.csv' if is_train else 'neural_network/test_result.csv'
        self.transform = transform
        self.is_train = is_train

        with open(fname, 'r') as f:
            self.data = [l.rstrip().split(',') for l in f.readlines()[1:]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        d = self.data[item]

        if self.is_train:
            fname = d[0]
            img = Image.open(fname)
        else:
            fname = os.path.join('testing_images', d[0])
            img = Image.open(fname)

        img = self.transform(img)

        shape = shape2label[d[1]]
        color = color2label[d[4]]

        x = float(d[3])
        if shape == 0:
            x -= 7

        y, x = float(d[2]) / 128, x / 128


        coords = torch.as_tensor([y, x]).float()

        return img, shape, color, coords, fname


# if __name__ == '__main__':
#     ds = Data(is_train=True)
#     dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=1)

#     for img, shape, color, coord, _ in dl:
#         plt.title(f'shape: {shape.item()}, '
#               f'color {color.item()}, '
#               f'x {coord[0][1] * 128}, '
#               f'y {coord[0][0] * 128}')

#         imshow(img[0], idx=0)



