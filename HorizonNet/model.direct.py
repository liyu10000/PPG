import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import functools


ENCODER_RESNET = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d'
]
ENCODER_DENSENET = [
    'densenet121', 'densenet169', 'densenet161', 'densenet201'
]


def lr_pad(x, padding=1):
    ''' Pad left/right-most to each other instead of zero padding '''
    return torch.cat([x[..., -padding:], x, x[..., :padding]], dim=3)


class LR_PAD(nn.Module):
    ''' Pad left/right-most to each other instead of zero padding '''
    def __init__(self, padding=1):
        super(LR_PAD, self).__init__()
        self.padding = padding

    def forward(self, x):
        return lr_pad(x, self.padding)


def wrap_lr_pad(net):
    for name, m in net.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if m.padding[1] == 0:
            continue
        w_pad = int(m.padding[1])
        m.padding = (m.padding[0], 0)
        names = name.split('.')
        root = functools.reduce(lambda o, i: getattr(o, i), [net] + names[:-1])
        setattr(
            root, names[-1],
            nn.Sequential(LR_PAD(w_pad), m)
        )


'''
Encoder
'''
class Resnet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(Resnet, self).__init__()
        assert backbone in ENCODER_RESNET
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        del self.encoder.fc, self.encoder.avgpool

    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x);
        x = self.encoder.layer2(x);
        x = self.encoder.layer3(x);
        x = self.encoder.layer4(x);
        return x


class Densenet(nn.Module):
    def __init__(self, backbone='densenet169', pretrained=True):
        super(Densenet, self).__init__()
        assert backbone in ENCODER_DENSENET
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        self.final_relu = nn.ReLU(inplace=True)
        del self.encoder.classifier

    def forward(self, x):
        lst = []
        for m in self.encoder.features.children():
            x = m(x)
            lst.append(x)
        features = [lst[4], lst[6], lst[8], self.final_relu(lst[11])]
        return features

    def list_blocks(self):
        lst = [m for m in self.encoder.features.children()]
        block0 = lst[:4]
        block1 = lst[4:6]
        block2 = lst[6:8]
        block3 = lst[8:10]
        block4 = lst[10:]
        return block0, block1, block2, block3, block4


'''
Decoder
'''
class ConvCompressH(nn.Module):
    ''' Reduce feature height by factor of two '''
    def __init__(self, in_c, out_c, ks=3):
        super(ConvCompressH, self).__init__()
        assert ks % 2 == 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=ks, stride=(2, 1), padding=ks//2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class GlobalHeightConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(GlobalHeightConv, self).__init__()
        self.layer = nn.Sequential(
            ConvCompressH(in_c, in_c//2),
            ConvCompressH(in_c//2, in_c//2),
            ConvCompressH(in_c//2, in_c//4),
            ConvCompressH(in_c//4, out_c),
        )

    def forward(self, x, out_w):
        x = self.layer(x)

        assert out_w % x.shape[3] == 0
        factor = out_w // x.shape[3]
        x = torch.cat([x[..., -1:], x, x[..., :1]], 3)
        x = F.interpolate(x, size=(x.shape[2], out_w + 2 * factor), mode='bilinear', align_corners=False)
        x = x[..., factor:-factor]
        return x


class GlobalHeightStage(nn.Module):
    def __init__(self, c1, c2, c3, c4, out_scale=8):
        ''' Process 4 blocks from encoder to single multiscale features '''
        super(GlobalHeightStage, self).__init__()
        self.cs = c1, c2, c3, c4
        self.out_scale = out_scale
        self.ghc_lst = nn.ModuleList([
            GlobalHeightConv(c1, c1//out_scale),
            GlobalHeightConv(c2, c2//out_scale),
            GlobalHeightConv(c3, c3//out_scale),
            GlobalHeightConv(c4, c4//out_scale),
        ])

    def forward(self, conv_list, out_w):
        assert len(conv_list) == 4
        bs = conv_list[0].shape[0]
        feature = torch.cat([
            f(x, out_w).reshape(bs, -1, out_w)
            for f, x, out_c in zip(self.ghc_lst, conv_list, self.cs)
        ], dim=1)
        return feature


'''
HorizonNet
'''
class HorizonNet(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.434, 0.447, 0.534])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.281, 0.267, 0.271])[None, :, None, None])

    def __init__(self, backbone, use_rnn):
        super(HorizonNet, self).__init__()
        self.backbone = backbone
        self.use_rnn = use_rnn
        self.out_scale = 8
        self.step_cols = 4
        self.rnn_hidden_size = 256

        # Encoder
        if backbone.startswith('res'):
            self.feature_extractor = Resnet(backbone, pretrained=True)
        elif backbone.startswith('dense'):
            self.feature_extractor = Densenet(backbone, pretrained=True)
        else:
            raise NotImplementedError()

        # 1D prediction
        c_last = 7680
        self.step_cols = 32
        self.linear = nn.Sequential(
            nn.Linear(c_last, self.rnn_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.rnn_hidden_size, 2 * self.step_cols),
        )
        # print(self.linear[-1].bias.data.shape)
        # self.linear[-1].bias.data[0*self.step_cols:1*self.step_cols].fill_(-1)
        # self.linear[-1].bias.data[1*self.step_cols:2*self.step_cols].fill_(-0.478)
        # self.linear[-1].bias.data[2*self.step_cols:3*self.step_cols].fill_(0.425)
        self.x_mean.requires_grad = False
        self.x_std.requires_grad = False
        wrap_lr_pad(self)

    def _prepare_x(self, x):
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x[:, :3] - self.x_mean) / self.x_std

    def forward(self, x):
        # if x.shape[2] != 512 or x.shape[3] != 1024:
        #     raise NotImplementedError()
        # print('x', x.shape)
        x = self._prepare_x(x)
        feature = self.feature_extractor(x)
        # print('feature', feature.shape)
        feature = feature.view(feature.shape[0], feature.shape[1]*feature.shape[2], feature.shape[3])
        feature = feature.permute(0, 2, 1)  # [b, w, c*h]
        # print('feature after permute', feature.shape)
        output = self.linear(feature)  # [b, w, 2 * step_cols]
        # print('output', output.shape)
        output = output.view(output.shape[0], output.shape[1], 2, self.step_cols)  # [b, w, 2, step_cols]
        # print('output expand view', output.shape)
        output = output.permute(0, 2, 1, 3)  # [b, 2, w, step_cols]
        # print('output after permute', output.shape)
        output = output.contiguous().view(output.shape[0], 2, -1)  # [b, 2, w*step_cols]
        # print('output contiguous view', output.shape)

        return output # B x 2 x W


if __name__ == '__main__':
    backbone = 'resnet34'
    dummy = torch.zeros(1, 3, 480, 640)

    # net = Resnet(backbone, pretrained=True)
    # out1, out2, out3, out4 = net(dummy)
    # print(out1.shape, out2.shape, out3.shape, out4.shape)

    net = HorizonNet(backbone, use_rnn=False)
    output = net(dummy)
    print(output.shape)
