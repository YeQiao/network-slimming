import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from models import *
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchsummary import summary
from helpernet import *
from utils import *
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix
import dill
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('log_dir')

def train(args, model, epoch, train_loader, optimizer, device, scaler):
    model.train()
        
    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate:', cur_lr)

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.to(device), target.to(device)
        # output = model(data)
        # loss = F.cross_entropy(output, target)
        # loss.backward()
        # optimizer.step()

        with autocast():
            output = model(data)
            loss = F.cross_entropy(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pred = output.data.max(1, keepdim=True)[1]
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(args, model, test_loader, device, scheduler = None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    if scheduler is not None:
        scheduler.step(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

def save_checkpoint(model, state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    # torch.save(model, os.path.join(filepath, 'model.pth.tar'), pickle_module=dill)

    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(os.path.join(filepath, 'model_scripted.pt')) # Save

    if is_best:
        shutil.copyfile(os.path.join(filepath, 'model_scripted.pt'), os.path.join(filepath, 'model_scripted_best.pt'))
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'checkpoint_best.pth.tar'))


def fine_tuning(args, model, train_loader,test_loader, device):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    scaler = GradScaler()
    best_prec1 = 0.
    for epoch in range(args.epochs):
        train(args, model, epoch, train_loader, optimizer, device, scaler)
        prec1 = test(args, model, test_loader, device, scheduler)
        is_best = prec1 > best_prec1
        best_model = model if is_best else best_model
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(model, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filepath=args.save)
    return best_model

def prune(args, model, test_loader, device):

    model = model.to(device)
    summary(model.cuda(), (3,32,32))

    # print(model)
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size
    # print('factor is : ', bn.numpy())
    # print('factor is : ', bn_avg)
    y, i = torch.sort(bn)
    thre_index = int(total * args.percent)
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    pruned_ratio = pruned/total

    print('Pre-processing Successful!')
    print('Start to make the real prune')

    acc = test(args, model, test_loader, device)

    # Make real prune
    print(cfg)
    print(len(cfg_mask))
    newmodel = vgg(dataset=args.dataset, cfg=cfg)
    if args.cuda:
        newmodel.cuda()

    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    savepath = os.path.join(args.save, "prune.txt")
    with open(savepath, "w") as fp:
        fp.write("Configuration: \n"+str(cfg)+"\n")
        fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
        fp.write("Test accuracy: \n"+str(acc))

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            print('Input channel index:', idx0)
            # print('weight shape: ', m0.weight.data.shape)
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

    # print(newmodel.modules())
    # layers = []
    # for m in newmodel.modules():
    #     if isinstance(m, nn.Conv2d):
    #         if m.in_channels != 0 and m.out_channels == 0:
    #             last_in_channel = m.in_channels
    #         elif m.in_channels == 0 and m.out_channels != 0:
    #             layers.append(nn.Conv2d(last_in_channel, last_out_channel, kernel_size=3, padding=1, bias=False))
    #         elif m.in_channels != 0 and m.out_channels != 0:
    #             layers.append(m)
    #     elif isinstance(m, nn.BatchNorm2d):
    #         if m.num_features == 0:
    #             continue
    #         else:
    #             layers.append(m)
    #     elif isinstance(m, nn.MaxPool2d):
    #         layers.append(m)
    #     elif isinstance(m, nn.AvgPool2d):
    #         layers.append(m)
    #     elif isinstance(m, nn.Linear):
    #         # print(m)
    #         layers.append(m)
    #     elif isinstance(m, nn.ReLU):
    #         layers.append(m)

    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()
            
        def forward(self, x):
            return x

    delete_layer = []
    last_in_channel = 0
    last_out_channel = newmodel.classifier.in_features
    ii = False
    for k, m in newmodel.named_modules():
        if type(m).__name__ == 'Conv2d':
            if m.in_channels != 0 and m.out_channels == 0:
                last_in_channel = m.in_channels
            if m.in_channels == 0 and m.out_channels != 0:
                replace_layer = k.split('.')
            elif m.in_channels == 0 or m.out_channels == 0:
                delete_layer.append(k.split('.'))
        elif type(m).__name__ == 'BatchNorm2d':
            if m.num_features == 0:
                delete_layer.append(k.split('.'))
                ii = True
            else:
                ii = False
        elif type(m).__name__ == 'ReLU' and ii:
            delete_layer.append(k.split('.'))

    newmodel.get_submodule(replace_layer[0])[int(replace_layer[1])] = nn.Conv2d(last_in_channel, last_out_channel, kernel_size=3, padding=1, bias=False)

    for i, (*parent, k) in enumerate(delete_layer):
        newmodel.get_submodule('.'.join(parent))[int(k)] = Identity()

    print(newmodel)
    model = newmodel.cuda()

    bn_avg = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_avg.append(m.weight.data.abs().mean())
    # print('factor is : ', bn.numpy())
    print('factor is : ', bn_avg)


    # test(model)
    summary(model.cuda(), (3,32,32))
    return model
    # torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))

def ws_quant(model, conv_bits, Linear_bits, device = "CUDA" if torch.cuda.is_available() else "cpu"):

    for k, m in model.named_modules():
        if type(m).__name__ == 'Conv2d':
            print("Quantizing conv layer : ", k)
            weight = m.weight.data.cpu().numpy()
            shape = weight.shape
            print("weight shape: " ,weight.shape)
            # mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
            mat = weight
            min_ = np.min(mat.data)
            max_ = np.max(mat.data)
            space = np.linspace(min_, max_, num=2**conv_bits)
            # kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, algorithm="full")
            # kmeans.fit(mat.data.reshape(-1,1))
            kmeans.fit(np.reshape(mat.data,(-1,1)))
            new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
            mat.data = new_weight
            m.weight.data = torch.from_numpy(mat).to(device)
        
        elif type(m).__name__ == 'Linear':
            print("Quantizing Linear layer")
            weight = m.weight.data.cpu().numpy()
            shape = weight.shape
            # mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
            mat = weight
            min_ = np.min(mat.data)
            max_ = np.max(mat.data)
            space = np.linspace(min_, max_, num=2**Linear_bits)
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, algorithm="full")
            # kmeans.fit(mat.data.reshape(-1,1))
            kmeans.fit(np.reshape(mat.data,(-1,1)))
            new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
            mat.data = new_weight
            m.weight.data = torch.from_numpy(mat).to(device)


def main():
    # Prune settings
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 512)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--depth', type=int, default=19,
                        help='depth of the vgg')
    parser.add_argument('--percent', type=float, default=0.5,
                        help='scale sparse rate (default: 0.5)')
    parser.add_argument('--model', default='', type=str, metavar='PATH',
                        help='path to the model (default: none)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save', default='./purned', type=str, metavar='PATH',
                        help='path to save prune model (default: ./purned)')
    parser.add_argument('--refine', action=argparse.BooleanOptionalAction, metavar='T',
                        help='weather to fine tuning the model')
    parser.add_argument('--quant', action=argparse.BooleanOptionalAction, metavar='T',
                        help='weather to quantize the model')
    parser.add_argument('--plot_hist', action=argparse.BooleanOptionalAction, metavar='T',
                        help='weather to plot the Histogram')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cpu'
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        device = 'cuda'

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model = vgg(dataset=args.dataset, depth=args.depth).to(device)

    if args.dataset == 'cifar10':
        data_loader = Cifar10DataLoader(batch_size = args.batch_size)
    else:
        data_loader = Cifar100DataLoader(batch_size = args.batch_size)

    train_loader = data_loader['train']
    test_loader = data_loader['val']

    if args.model:
        if os.path.isfile(args.model):
            print("=> loading checkpoint '{}'".format(args.model))
            checkpoint = torch.load(args.model)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                .format(args.model, checkpoint['epoch'], best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.model))

    test(args, model, test_loader, device)

    print('Start Model Purning')
    model = prune(args, model, test_loader, device)
    print('Model Purning Successful!')
    best_model = model
    if args.refine:
        print('----------Start Fine Tuning-----------')
        best_model = fine_tuning(args, model, train_loader,test_loader, device)
        test(args, best_model, test_loader, device)
        print('----------Fine Tuning Finished-----------')

    if args.quant:
        print('----------Start Quantization-----------')
        ws_quant(best_model, 4, 5, device)
        test(args, best_model, test_loader, device)
        print('----------Quantization Finished-----------')
        
    if args.plot_hist:
        plot_hist_conv_linear(best_model,save_fig=True,plt_show=True,model_name=None)

if __name__ == "__main__":
    main()