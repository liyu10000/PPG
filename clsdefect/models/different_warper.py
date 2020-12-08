from collections import OrderedDict
import copy
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Warp(nn.Module):
    def __init__(self, patch_size: Tuple[int, int]):
        super().__init__()
        # self.uniform_sampler = F.affine_grid(torch.eye(2, 3).unsqueeze(0), size=(1, 1, *patch_size))
        # self.flow_generator = nn.Sequential(
        #     nn.Conv2d(3, 100, 7, 1, 3),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(100, 100, 5, 1, 2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(100, 50, 3, 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(50, 2, 1, 1, 0),
        #     nn.Tanh(),
        # )
        self.patch_size = patch_size
        self.theta_generator = nn.Sequential(
            nn.Conv2d(3, 100, 7, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(100, 100, 5, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(100, 50, 3, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(50, 6, 1, 1, 0),
            nn.AdaptiveAvgPool2d(1),
        )

    def grid_generator(self, theta, batch_size: int):
        return F.affine_grid(theta, size=(batch_size, 1, self.patch_size[0], self.patch_size[1]), align_corners=False)

    def forward(self, image):
        # flow = self.flow_generator(image)
        # flow = self.mesh + flow
        # return F.grid_sample(image, flow, mode='bilinear', padding_mode='border')
        N = image.shape[0]
        theta = self.theta_generator(image).view(N, 2, 3)
        grid = self.grid_generator(theta, N)
        return F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=False), theta


class Model(nn.Module):
    def __init__(self, num_classes, params="", pretrained=True):
        super(Model, self).__init__()
        params = params.split('_')
        assert len(params) == 2 or len(params) == 3
        H, W = [int(param) for param in params[:2]]
        if len(params) == 3:
            dropRate = float(params[-1])
        else:
            dropRate = 0.

        classifier = models.densenet121(pretrained=pretrained)
        self.features = copy.deepcopy(classifier.features)
        self.special_features = copy.deepcopy(models.densenet121(pretrained=False).features)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.warp = Warp((H, W))

        self.classifier_head = nn.Sequential(OrderedDict([
            ('L1', nn.Linear(1024, 512)),
            ('actv1', nn.ReLU(inplace=True)),
            ('drop1', nn.Dropout(p=dropRate)),
            ('L2', nn.Linear(512, 256)),
            ('actv2', nn.ReLU(inplace=True)),
            ('drop2', nn.Dropout(p=dropRate)),
        ]))
        self.delam_head = nn.Sequential(OrderedDict([
            ('L1', nn.Linear(2*1024, 512)),
            ('actv1', nn.ReLU(inplace=True)),
            ('drop1', nn.Dropout(p=dropRate)),
            ('L2', nn.Linear(512, 256)),
            ('actv2', nn.ReLU(inplace=True)),
            ('drop2', nn.Dropout(p=dropRate)),
        ]))
        self.classifiers = nn.ModuleList([nn.Sequential(OrderedDict([
            ('L3', nn.Linear(256, 128)),
            ('actv3', nn.ReLU(inplace=True)),
            ('drop3', nn.Dropout(p=dropRate)),
            ('L4', nn.Linear(128, 1)),
        ])) for _ in range(num_classes)])  # building multiple binary classifiers for each class
        self.actv = nn.Sigmoid()

    def freeze(self, what='features'):
        if what == 'features':
            print("Freezing features")
            for p in self.features.parameters():
                p.requires_grad = False
        
    def forward(self, patches, **inputs):
        warped_patches_1, flow_1 = self.warp(patches)
        warped_patches_2, flow_2 = self.warp(patches)
        features = self.avg_pool(self.features(warped_patches_1)).squeeze(dim=2).squeeze(dim=2)
        delam_feats = self.avg_pool(self.special_features(warped_patches_2)).squeeze(dim=2).squeeze(dim=2)
        delam_head = self.delam_head(torch.cat((features, delam_feats), dim=1))
        classifier_head = self.classifier_head(features)
        out = []
        for classifier in self.classifiers[:-1]:
            out.append(classifier(classifier_head))

        out.append(self.classifiers[-1](delam_head))

        out = torch.cat(out, dim=1)
        output = {
            'raw': out,
            'pmf': self.actv(out),
            'features': {
                'general': features,
                'delam': delam_feats,
            },
            # 'flow': flow,
            # 'warped': warped_patches,
        }
        return output

    def hook_maps(self, hooker):
        return self.avg_pool.register_forward_hook(hooker)

    def extract_weights_hook(self):
        return self.parameters()