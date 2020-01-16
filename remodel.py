"""
Wide ResNet by Sergey Zagoruyko and Nikos Komodakis
Fixup initialization by Hongyi Zhang, Yann N. Dauphin, Tengyu Ma
Based on code by xternalz and Andy Brock:
https://github.com/xternalz/WideResNet-pytorch
https://github.com/ajbrock/BoilerPlate
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    droprate = 0.0
    use_bn = True
    use_fixup = False
    fixup_l = 12
    const = 1.0
    lrelu = 0.0
    def __init__(self, in_planes, out_planes, stride,i):



        super(BasicBlock, self).__init__()


        if i % 2:
            phase = -1
        else:
            phase = 1
        ##print("Use fixup:")
        ##print(self.use_fixup)
        
        self.bn = nn.BatchNorm2d(out_planes)

        if self.lrelu ==0.0:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.LeakyReLU(self.lrelu)


        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.equalInOut = in_planes == out_planes
        assert (
            self.use_fixup or self.use_bn
        ), "Need to use at least one thing: Fixup or BatchNorm"

        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(2)]
        )

        k = (
            self.conv.kernel_size[0]
            * self.conv.kernel_size[1]
            * self.conv.out_channels
        )

##        print("phase: %d, equal: %s " % ( phase , self.equalInOut))

        if self.equalInOut:
            ConstIdentity(self.conv.weight, self.conv.bias, torch.nn.init.calculate_gain('relu'), self.const, phase, self.lrelu)
        else:
            ConstDeltaOrthogonal(self.conv.weight, self.conv.bias, torch.nn.init.calculate_gain('relu'),self.const)



    def forward(self, x):


        out = self.relu(self.conv(x))


        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        if self.use_bn:
            out = self.bn(out)

        return out

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride):
        layers = []

        for i in range(int(nb_layers)):
            _in_planes = i == 0 and in_planes or out_planes
            _stride = i == 0 and stride or 1
            layers.append(block(_in_planes, out_planes, _stride,i))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class NarrowNet(nn.Module):
    def __init__(
        self,
        depth,
        num_classes,
        widen_factor=1,
        droprate=0.0,
        use_fixup=False,
        use_bn=True,
        varnet=False,
        noise=0.0,
        lrelu=0.0,
    ):
        super(NarrowNet, self).__init__()

        if varnet:
            noise=1.0


        ##nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        nChannels = [widen_factor, widen_factor , widen_factor , widen_factor]


        assert (depth - 4) % 3 == 0, "You need to change the number of layers"
        n = (depth - 4) / 3

        BasicBlock.droprate = droprate
        BasicBlock.use_bn = use_bn
        BasicBlock.fixup_l = n * 3
        BasicBlock.use_fixup = use_fixup
        BasicBlock.const = 1.0-noise
        BasicBlock.lrelu = lrelu


        ##print("Use fixup WideResnet:")
        ##print(use_fixup)
        ##print("Use BN WideResnet:")
        ##print(use_bn)
        block = BasicBlock

        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        k = (
            self.conv1.kernel_size[0]
            * self.conv1.kernel_size[1]
            * self.conv1.out_channels
        )
        
        ConstDeltaOrthogonal(self.conv1.weight, self.conv1.bias, torch.nn.init.calculate_gain('relu'), 1.0 - noise)

        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2)

        if lrelu ==0.0:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.LeakyReLU(lrelu)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        if noise > 0.0:
            self.fc.weight.data.normal_(0, math.sqrt(noise / k))
        else:
            self.fc.weight.data.zero_()
        
        self.fc.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):


        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)




    def misc(self,x):
        h1 = self.conv1(x)
        h2 = self.block1(h1)
        h3 = self.block2(h2)
        h4 = self.block3(h3)
        h45 = self.relu(h4)
        h5 = F.avg_pool2d(h45, 8)
        h6 = h5.view(-1, self.nChannels)
        return self.fc(h6), h1, h2, h3, h4, h5, h6




class ConvNet(nn.Module):
    def __init__(
        self,
        depth,
        num_classes,
        widen_factor=1,
        droprate=0.0,
        use_fixup=False,
        use_bn=True,
        varnet=False,
        noise=0.0,
        lrelu=0.0,
    ):
        super(ConvNet, self).__init__()

        if varnet:
            noise=1.0

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        assert (depth - 4) % 3 == 0, "You need to change the number of layers"
        n = (depth - 4) / 3

        BasicBlock.droprate = droprate
        BasicBlock.use_bn = use_bn
        BasicBlock.fixup_l = n * 3
        BasicBlock.use_fixup = use_fixup
        BasicBlock.const = 1.0-noise
        BasicBlock.lrelu = lrelu


        ##print("Use fixup WideResnet:")
        ##print(use_fixup)
        ##print("Use BN WideResnet:")
        ##print(use_bn)
        block = BasicBlock

        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        k = (
            self.conv1.kernel_size[0]
            * self.conv1.kernel_size[1]
            * self.conv1.out_channels
        )
        
        ConstDeltaOrthogonal(self.conv1.weight, self.conv1.bias, torch.nn.init.calculate_gain('relu'), 1.0 - noise)

        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2)

        if lrelu ==0.0:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.LeakyReLU(lrelu)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        if noise > 0.0:
            self.fc.weight.data.normal_(0, math.sqrt(noise / k))
        else:
            self.fc.weight.data.zero_()
        
        self.fc.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward2(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

    def forward(self,x):
        h1 = self.conv1(x)
        h2 = self.block1(h1)
        h3 = self.block2(h2)
        h4 = self.block3(h3)
        h45 = self.relu(h4)
        h5 = F.avg_pool2d(h45, 8)
        h6 = h5.view(-1, self.nChannels)
        return self.fc(h6), h1, h2, h3, h4, h5, h6



def genOrthgonal(dim):
    a = torch.zeros((dim, dim)).normal_(0, 1)
    q, r = torch.qr(a)
    d = torch.diag(r, 0).sign()
    diag_size = d.size(0)
    d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
    q.mul_(d_exp)
    return q

def makeLambdaDeltaOrthogonal(weights, bias, gain):
    rows = weights.size(0)
    cols = weights.size(1)
    if bias is not None:
        nn.init.constant_(bias, 0)
    dim = max(rows, cols)
    mid1 = weights.size(2) // 2
    mid2 = weights.size(3) // 2
    nn.init.constant_(weights, 0)
    weights.data[:, :, mid1, mid2] = gain ##torch.ones([rows, cols]) * gain

def ConstIdentity(weights, bias, gain, const = 1.0, phase = 1, lrelu = 0.0):

    rows = weights.size(0)
    cols = weights.size(1)
 
    assert(rows==cols)

    if const < 1.0:
        k = cols * weights.size(2) * weights.size(3)
        weights.data.normal_(0, math.sqrt(gain*(1.0-const) / k))
    else: 
        nn.init.constant_(weights, 0)

    if bias is not None:
        nn.init.constant_(bias, 0)

    if const > 0.0:
        mid1 = weights.size(2) // 2
        mid2 = weights.size(3) // 2
        factor = phase*(1.0/cols**2)*const
        if phase == -1 and lrelu > 0.0:
            factor = factor/lrelu

        for d0 in range(rows):
            weights.data[:, :, mid1, mid2] += factor


def ConstDeltaOrthogonal(weights, bias, gain, const = 1.0):

    rows = weights.size(0)
    cols = weights.size(1)
 
    if const < 1.0:
        k = cols * weights.size(2) * weights.size(3)
        weights.data.normal_(0, math.sqrt(gain*(1.0-const) / k))

    else: 
        nn.init.constant_(weights, 0)


    if bias is not None:
        nn.init.constant_(bias, 0)

    if const > 0.0:
        mid1 = weights.size(2) // 2
        mid2 = weights.size(3) // 2
        weights.data[:, :, mid1, mid2] = gain/cols ##torch.ones([rows, cols]) * gain      
