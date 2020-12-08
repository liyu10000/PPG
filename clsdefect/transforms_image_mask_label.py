import random
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import torch


class Compose(T.Compose):
    def __init__(self, transforms):
        super(Compose, self).__init__(transforms)

    def __call__(self, image, label, mask=None):
        for t in self.transforms:
            image, label, mask = t(image, label, mask)
        return image, label, mask


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, p):
        super(RandomHorizontalFlip, self).__init__(p)

    def __call__(self, image, label, mask=None):
        if random.random() < self.p:
            image = F.hflip(image)
            if mask: mask = F.hflip(mask)
        return image, label, mask


class RandomVerticalFlip(T.RandomVerticalFlip):
    def __init__(self, p):
        super(RandomVerticalFlip, self).__init__(p)

    def __call__(self, image, label, mask=None):
        if random.random() < self.p:
            image = F.vflip(image)
            if mask: mask = F.vflip(mask)
        return image, label, mask


class ToTensor(T.ToTensor):
    def __call__(self, image, label, mask=None):
        if mask:
            return super(ToTensor, self).__call__(image), label, super(ToTensor, self).__call__(mask)
        else:
            return super(ToTensor, self).__call__(image), label, None


class ToTensorEncodeLabels(T.ToTensor):
    def __init__(self, num_classes):
        super(ToTensorEncodeLabels, self).__init__()
        self.num_classes = num_classes

    def encode_labels(self, indices):
        labels = torch.zeros(self.num_classes)
        labels[indices] = 1
        return labels

    def __call__(self, image, label_indices, mask=None):
        if mask is not None:
            return super(ToTensorEncodeLabels, self).__call__(image), self.encode_labels(label_indices), super(ToTensorEncodeLabels, self).__call__(mask).type(torch.LongTensor)
        else:
            return super(ToTensorEncodeLabels, self).__call__(image), self.encode_labels(label_indices), None


class DecodeLabels(object):
    def __init__(self, num_classes):
        super(DecodeLabels, self).__init__()
        self.num_classes = num_classes

    def decode_labels(self, labels):
        if len(labels.shape)<2:
            labels = labels.unsqueeze(0)
        assert labels.shape[1] == self.num_classes
        indices = []
        for ib in range(labels.shape[0]):
            indices.append((labels[ib] >= 0.5).nonzero().squeeze().tolist())
        return indices

    def __call__(self, labels):
        return self.decode_labels(labels)


class Normalize(T.Normalize):
    def __init__(self, mean, std, inplace=False):
        super(Normalize, self).__init__(mean, std, inplace)

    def __call__(self, image, label, mask=None):
        return super(Normalize, self).__call__(image), label, mask



class Resize(T.Resize):
    def __init__(self, size, interpolation=Image.BILINEAR):
        super(Resize, self).__init__(size, interpolation)

    def __call__(self, image, label, mask=None):
        if mask:
            return super(Resize, self).__call__(image), label, super(Resize, self).__call__(mask)
        else:
            return super(Resize, self).__call__(image), label, None


class CenterCrop(T.CenterCrop):
    def __init__(self, size):
        super(CenterCrop, self).__init__(size)

    def __call__(self, image, label, mask=None):
        if mask:
            return super(CenterCrop, self).__call__(image), label, super(CenterCrop, self).__call__(mask)
        else:
            return super(CenterCrop, self).__call__(image), label, None


class Pad(T.Pad):
    def __init__(self, **kwargs):
        super(Pad, self).__init__(**kwargs)

    def __call__(self, image, label, mask=None):
        if mask:
            return super(Pad, self).__call__(image), label, super(Pad, self).__call__(mask)
        else:
            return super(Pad, self).__call__(image), label, None
