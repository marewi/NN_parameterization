import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from progress.bar import Bar
import matplotlib.pyplot as plt

from neural_network.model import Net
from neural_network.loss import Loss, AverageMeter
from neural_network.dataset import Data, imshow, label2color, label2shape


# trained model path
MODEL_PATH = 'model.pth.tar'

def write_to_file(text, file):
    with open(file, 'a') as f:
        f.write(f'{text}\n')

def save_plots(plot_dict):
    os.makedirs('plots', exist_ok=True)
    for k,v in plot_dict.items():
        plt.title(f'Training plot for \"{k}\" loss')
        plt.plot(v)
        plt.savefig(f'plots/{k}_loss_plot.png')
        # plt.show()
        plt.cla()


# Training of NN
def train(end_epoch, batch_size, learning_rate):
    # print(f'training running with end_epoch = {end_epoch} & batch_size = {batch_size} & learning_rate = {learning_rate}')

    # end_epoch = 20
    # batch_size = 16
    # learning_rate = 0.001
    n_workers = 8

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Creating NN Model')
    net = Net().to(device)

    train_ds = Data(is_train=True)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    criterion = Loss()

    plot_dict = {
        'total': [],
        'shape': [],
        'color': [],
        'coord': [],
    }

    print('Started NN Training')
    for epoch in range(end_epoch):

        losses = AverageMeter()

        # bar = Bar(f'Epoch {epoch + 1}/{end_epoch}', fill='#', max=len(train_dl))

        for i, data in enumerate(train_dl, 0):

            img, shape, color, coords, _ = data

            img = img.to(device)
            shape = shape.to(device)
            color = color.to(device)
            coords = coords.to(device)

            optimizer.zero_grad()

            outputs = net(img)
            labels = [shape, color, coords]
            loss, loss_dict = criterion(outputs, labels)
            losses.update(loss.item(), img.size(0))
            loss.backward()
            optimizer.step()

            plot_dict['total'].append(loss.item())
            for k, v in loss_dict.items():
                plot_dict[k].append(v.item())

            # summary_string = f'({i + 1}/{len(train_dl)}) | Total: {bar.elapsed_td} | ' \
            #                  f'ETA: {bar.eta_td:} | loss: {losses.avg:.4f}'

            # for k, v in loss_dict.items():
            #     summary_string += f' | {k}: {v:.4f}'

        #     bar.suffix = summary_string
        #     bar.next()

        # bar.finish()

    # print('Finished Training')

    # print('loss_avg: {losses.avg:.4f}')

    # print('Saving model')

    # save_plots(plot_dict)

    # torch.save(net.state_dict(), MODEL_PATH)

    return(losses.avg)


# # Testing of NN
# def test():
#     test_ds = Data(is_train=False)
#     test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=1)

#     print('Creating Model')
#     net = Net()
#     net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     net = net.to(device).eval()

#     result_filename = 'predicted_test_result.csv'
#     write_to_file(text='filename,logo-name,x,y,color', file=result_filename)

#     color_acc, shape_acc, coord_err = [], [], []

#     print('Started Testing')
#     with torch.no_grad():
#         bar = Bar(f'Test', fill='#', max=len(test_dl))

#         for i, data in enumerate(test_dl, 0):
#             img, shape, color, coord, fname = data

#             img = img.to(device)
#             shape = shape.to(device)
#             color = color.to(device)
#             coord = coord.to(device) * 128

#             output = net(img)

#             pshape, pcolor, pcoord = output

#             _, pshape = torch.max(pshape, 1)
#             _, pcolor = torch.max(pcolor, 1)
#             pcoord *= 128

#             img = img[0]

#             pshape, pcolor, pcoord = pshape[0], pcolor[0], pcoord[0]
#             tshape, tcolor, tcoord = shape[0], color[0], coord[0]

#             pcoord = pcoord.cpu().numpy()
#             tcoord = tcoord.cpu().numpy()

#             pshape, tshape = label2shape[int(pshape)], label2shape[int(tshape)]
#             pcolor, tcolor = label2color[int(pcolor)], label2color[int(tcolor)]

#             color_acc.append(pcolor == tcolor)
#             shape_acc.append(pshape == tshape)
#             coord_err.append(np.absolute(pcoord - tcoord))

#             title = f'Predicted shape: {pshape}, color: {pcolor}, ' \
#                     f'coord: ({int(pcoord[1].round())}/{int(pcoord[0].round())})\n'

#             title += f'True shape: {tshape}, color: {tcolor}, ' \
#                      f'coord: ({int(tcoord[1])}/{int(tcoord[0])})'

#             # print 20 sample images
#             if i < 20:
#                 plt.title(title)
#                 imshow(img, idx=i)

#             ## code for writing to csv and than to xlsx
#             fname = fname[0]
#             write_to_file(f'{fname},'
#                           f'{pshape},'
#                           f'{int(pcoord[1].round())},'
#                           f'{int(pcoord[0].round())},'
#                           f'{pcolor}',
#                           result_filename)

#             summary_string = f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'

#             bar.suffix = summary_string
#             bar.next()

#         bar.finish()

#         read_file = pd.read_csv(result_filename)
#         read_file.to_excel('predicted_test_result.xlsx', index = None, header=True)

#         # mean error in pixels
#         coord_err = np.array(coord_err)
#         mean_error_y = coord_err[0].mean()
#         mean_error_x = coord_err[1].mean()

#         # std of error
#         std_error_y = coord_err[0].std()
#         std_error_x = coord_err[1].std()

#         print(f'Mean pixel error x: {mean_error_x:.4f}, y: {mean_error_y:.4f}')
#         print(f'Std of error x: {std_error_x:.4f}, y: {std_error_y:.4f}')

#         # color accuracy
#         color_acc = np.array(color_acc)
#         color_acc = (color_acc.sum() / color_acc.shape[0]) * 100
#         print(f'Color accuracy: {color_acc:.4f}')

#         # shape accuracy
#         shape_acc = np.array(shape_acc)
#         shape_acc = (shape_acc.sum() / shape_acc.shape[0]) * 100
#         print(f'Shape accuracy: {shape_acc:.4f}')

# if __name__ == '__main__':
#     train()
#     # test()

