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


def train(args, model, epoch, train_loader, optimizer, device, scaler):
    model.train()
        
    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate:', cur_lr)

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.to(device), target.to(device)
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

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

def fine_tuning(args, model, train_loader,test_loader, device):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    scaler = GradScaler()
    best_prec1 = 0.
    for epoch in range(args.epochs):
        train(args, model, epoch, train_loader, optimizer, device, scaler)
        prec1 = test(args, model, test_loader, device, scheduler)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filepath=args.save)


def ws_quant(model, conv_bits, Linear_bits, device = "CUDA" if torch.cuda.is_available() else "cpu"):

    for m in model.named_modules():
        if type(m).__name__ == 'Conv2d':
            weight = m.weight.data.cpu().numpy()
            shape = weight.shape
            mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
            min_ = min(mat.data)
            max_ = max(mat.data)
            space = np.linspace(min_, max_, num=2**conv_bits)
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
            kmeans.fit(mat.data.reshape(-1,1))
            new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
            mat.data = new_weight
            m.weight.data = torch.from_numpy(mat.toarray()).to(device)
        
        elif type(m).__name__ == 'Linear':
            weight = m.weight.data.cpu().numpy()
            shape = weight.shape
            mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
            min_ = min(mat.data)
            max_ = max(mat.data)
            space = np.linspace(min_, max_, num=2**Linear_bits)
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
            kmeans.fit(mat.data.reshape(-1,1))
            new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
            mat.data = new_weight
            m.weight.data = torch.from_numpy(mat.toarray()).to(device)


def main():
    # Prune settings
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 512)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
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
    parser.add_argument('--quant', action=argparse.BooleanOptionalAction, metavar='T',
                        help='weather to fine tuning the model')
    parser.add_argument('--conv_bits', type=int, default=4,
                        help='bit width for conv layer')
    parser.add_argument('--Linear_bits', type=int, default=5,
                        help='bit width for Linear layer')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cpu'
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        device = 'cuda'

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.dataset == 'cifar10':
        data_loader = Cifar10DataLoader(batch_size = args.batch_size)
    else:
        data_loader = Cifar100DataLoader(batch_size = args.batch_size)

    train_loader = data_loader['train']
    test_loader = data_loader['val']

    if args.model:
        if os.path.isfile(args.model):
            model = torch.load(args.model).to(device)
            model.eval()
            print("=> loaded model")
        else:
            print("=> no model found at '{}'".format(args.model))
    else:
        raise Exception("Sorry, no valid model path was provided")

    print("=> original ACC: '{}'".format(test(args, model, test_loader, device)))

    if args.refine:
        print('----------Start Fine Tuning-----------')
        fine_tuning(args, model, train_loader,test_loader, device)
        print('----------Fine Tuning Finished-----------')

    if args.quant:
        print('----------Start Quantization-----------')
        ws_quant(model, args.conv_bits, args.Linear_bits, device)
        print('----------Quantization Finished-----------')
        print("=> quantized ACC: '{}' with conv_bits: '{}', Linear_bits: '{}'".format(test(args, model, test_loader, device), args.conv_bits, args.Linear_bits))



if __name__ == "__main__":
    main()