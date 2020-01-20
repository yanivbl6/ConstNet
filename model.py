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



class Swish(nn.Module):

    def forward(self, input):
        return (input + torch.sigmoid(input))

    def __repr__(self):
        return self.__class__.__name__ + '  ()'

class BasicBlock(nn.Module):
    droprate = 0.0
    use_bn = True
    use_fixup = False
    fixup_l = 12
    varnet = False
    def __init__(self, in_planes, out_planes, stride):
        super(BasicBlock, self).__init__()

        
        ##print("Use fixup:")
        ##print(self.use_fixup)
        
        self.bn = nn.BatchNorm2d(in_planes)
##        self.relu = nn.Softplus()
##        self.relu = Swish()
        self.relu = nn.ReLU(inplace=True)

        gain = torch.nn.init.calculate_gain('relu')
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.equalInOut = in_planes == out_planes

        self.conv_res = nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
        )
        self.conv_res = not self.equalInOut and self.conv_res or None
        self.no_act =  not self.equalInOut
            
        assert (
            self.use_fixup or self.use_bn
        ), "Need to use at least one thing: Fixup or BatchNorm"

        if self.use_fixup:

            self.scale = nn.Parameter(torch.ones(1))
            self.biases = nn.ParameterList(
                [nn.Parameter(torch.zeros(1)) for _ in range(2)]
            )

            k = (
                self.conv.kernel_size[0]
                * self.conv.kernel_size[1]
                * self.conv.out_channels
            )

            if self.varnet:
                self.conv.weight.data.normal_(
                    0, self.fixup_l ** (-0.5) * gain * math.sqrt(1.0 / k)
                )
            else:
                self.conv.weight.data.zero_()

            if self.conv_res is not None:
                k = (
                    self.conv_res.kernel_size[0]
                    * self.conv_res.kernel_size[1]
                    * self.conv_res.out_channels
                )
                if self.varnet:
                    self.conv_res.weight.data.normal_(0, gain * math.sqrt(1.0 / k))
                else:
                    self.conv_res.weight.data.fill_(gain**2 / k)

    def forward(self, x):
        if self.use_bn:
            x_out = self.relu(self.bn(x))
            out = self.conv(x_out)
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
        else:
            x_out = self.relu(x + self.biases[0])
            out = self.conv(x_out) + self.biases[1]

##            x_out = self.relu(x + self.biases[0])
##            out = self.conv(x_out) + self.biases[1]

            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
            out = self.scale * out

        if self.equalInOut:
            return torch.add(x, out)

        return torch.add(self.conv_res(x_out), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride):
        layers = []

        for i in range(int(nb_layers)):
            _in_planes = i == 0 and in_planes or out_planes
            _stride = i == 0 and stride or 1
            layers.append(block(_in_planes, out_planes, _stride))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class ResNet5(nn.Module):
    def __init__(
        self,
        depth,
        num_classes,
        widen_factor=1,
        droprate=0.0,
        use_bn=True,
        use_fixup=False,
        varnet=False,
        noise=0.0,
        lrelu=0.0,
    ):
        super(ResNet5, self).__init__()

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor, 128 * widen_factor, 256 * widen_factor]

        assert (depth - 4) % 5 == 0, "You need to change the number of layers"
        n = (depth - 4) / 5

        BasicBlock.droprate = droprate
        BasicBlock.use_bn = use_bn
        BasicBlock.fixup_l = n * 3
        BasicBlock.use_fixup = use_fixup
        BasicBlock.varnet = varnet

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
        gain = torch.nn.init.calculate_gain('relu')
        if varnet:
            self.conv1.weight.data.normal_(0, math.sqrt(2.0 / k))
        else:
            makeLambdaDeltaOrthogonal(self.conv1.weight, self.conv1.bias, gain ) ##*torch.nn.init.calculate_gain('relu'))
        
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2)
        self.block4 = NetworkBlock(n, nChannels[3], nChannels[4], block, 2)
        self.block5 = NetworkBlock(n, nChannels[4], nChannels[5], block, 2)



        self.relu = nn.ReLU(inplace=True)
##        self.relu = nn.Softplus()
##        self.relu = Swish()
        self.fc = nn.Linear(nChannels[5], num_classes)
        self.nChannels = nChannels[5]

        self.fc.bias.data.zero_()
        self.fc.weight.data.zero_()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        out = self.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)



class WideResNet(nn.Module):
    def __init__(
        self,
        depth,
        num_classes,
        widen_factor=1,
        droprate=0.0,
        use_bn=True,
        use_fixup=False,
        varnet=False,
        noise=0.0,
        lrelu=0.0,
    ):
        super(WideResNet, self).__init__()

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        assert (depth - 4) % 3 == 0, "You need to change the number of layers"
        n = (depth - 4) / 3

        BasicBlock.droprate = droprate
        BasicBlock.use_bn = use_bn
        BasicBlock.fixup_l = n * 3
        BasicBlock.use_fixup = use_fixup
        BasicBlock.varnet = varnet

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
        
        gain = torch.nn.init.calculate_gain('relu')
        if varnet:
            self.conv1.weight.data.normal_(0,  gain * math.sqrt(1.0 / k))
        else:
            makeLambdaDeltaOrthogonal(self.conv1.weight, self.conv1.bias, gain) 
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2)

        self.relu = nn.ReLU(inplace=True)

##        self.relu = nn.Softplus()
##        self.relu = Swish()

        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.fc.bias.data.zero_()
        self.fc.weight.data.zero_()

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
