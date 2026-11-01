"""VGG6 for CIFAR-10 (professor's template) with a configurable activation function."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CFG_VGG6 = [64, 64, 'M', 128, 128, 'M']

# Name -> activation class. Q2(a) asks for ReLU, Sigmoid, Tanh, SiLU, GELU.
ACTIVATIONS = {
    'relu': nn.ReLU(inplace=True),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'silu': nn.SiLU(inplace=True),
    'gelu': nn.GELU()


}


def make_layers(cfg, activation="relu", batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            act = ACTIVATIONS.get(activation.lower(), nn.ReLU(inplace=True))
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), act]
            else:
                layers += [conv2d, act]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(128, num_classes))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)                     # (N, 128, 8, 8)
        x = F.adaptive_avg_pool2d(x, (1, 1))     # (N, 128, 1, 1)  average each channel
        x = torch.flatten(x, 1)                  # (N, 128)
        return self.classifier(x)                # (N, 10)  one score per class

    def _initialize_weights(self):
        # Same initialisation as the professor's template
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def vgg6(activation="relu", batch_norm=True, num_classes=10):
    return VGG(make_layers(CFG_VGG6, activation, batch_norm), num_classes)
