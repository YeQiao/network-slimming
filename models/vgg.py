import math
from unittest import skip
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
__all__ = ['vgg']

defaultcfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class quantized_conv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, bits = 5):
        super(quantized_conv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.out_chn = out_chn
        self.conv_inf = nn.Conv2d(in_chn, out_chn, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bits = bits

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        shape = self.shape
        mat = csr_matrix(real_weights) if shape[0] < shape[1] else csc_matrix(real_weights)
        min_ = min(mat.data)
        max_ = max(mat.data)
        space = np.linspace(min_, max_, num=2**self.bits) 
        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
        kmeans.fit(mat.data.reshape(-1,1))
        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        mat.data = new_weight

        if self.training:
            y = F.conv2d(x, new_weight, stride=self.stride, padding=self.padding)
        else:
            # self.conv_inf.weight.data = torch.from_numpy(mat.toarray()).to("cuda")
            self.conv_inf.weight = torch.nn.Parameter(torch.from_numpy(mat.toarray()), requires_grad=False)
            y = self.conv_inf(x)
        return y

class vgg(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.feature = self.make_layers(cfg, True)
        self.feature.append(nn.AvgPool2d(2))

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        self.classifier = nn.Linear(cfg[-1], num_classes)
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

        

    def forward(self, x):
        x = self.feature(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                if n==0:
                    continue
                else:
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__ == '__main__':
    net = vgg()
    x = Variable(torch.FloatTensor(16, 3, 40, 40))
    y = net(x)
    print(y.data.shape)