import torch
from torch import nn
from torchvision.transforms import ToTensor
import os
import numpy as np
import sys
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, dimensions, hidden_dim, number_of_classes, multilabel_bool):
        super().__init__()
        self.dimensions = dimensions
        self.hidden_dim = hidden_dim
        self.number_of_classes = number_of_classes
        self.multilabel_bool = multilabel_bool

      #  MLP classification layer
        self.classifier = nn.Sequential(
            nn.Linear(self.dimensions, hidden_dim),
            nn.Dropout(0.30, inplace = True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.30, inplace = True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, number_of_classes)
        )
   
    def forward(self, x):
        x = x.squeeze(1)
        out = torch.sigmoid(self.classifier(x))        
        return out