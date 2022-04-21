import torch.nn as nn
from collections import OrderedDict
import math

FEATURE_SIZE = 310
OUTPUT_SIZE = 3


class BackboneNetwork(nn.Module):
    def __init__(self):
        super(BackboneNetwork, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(FEATURE_SIZE, 256)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(256, 128)),
            ('relu2', nn.ReLU())
        ]))

    def forward(self, x):
        return self.model(x)


class Classifier(nn.Module):
    def __init__(self, class_num=OUTPUT_SIZE, extract=True, dropout_p=0.5):
        super(Classifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )
        self.fc2 = nn.Linear(64, class_num)
        self.extract = extract
        self.dropout_p = dropout_p

    def forward(self, x):
        fc1_emb = self.fc1(x)
        if self.training:
            fc1_emb.mul_(math.sqrt(1-self.dropout_p))
        logit = self.fc2(fc1_emb)

        if self.extract:
            return fc1_emb, logit
        return logit
