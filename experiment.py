import torch
import numpy as np
from attrdict import AttrDict
from monomial_neural_network import *

default_args = AttrDict(
    {
    'teacher_W': np.array([[1,2, 0.4],[1,-1, -0.2],[0,2, 0.9], [1,1, 0]]),
    'teacher_v': np.array([[0, 1, 1, -0.2]]),
    'dim_x': 2,
    'dim_y': 1,
    'layers': [50, 50],
    'data': 10,
    'num_epochs': 1000,
    'lr': 0.01
    }
)

class Experiment:
    def __init__(self, args = AttrDict()):
        self.args = default_args + args
        self.teacher_W = self.args.teacher_W
        self.teacher_v = self.args.teacher_v
        self.dim_x = self.args.dim_x
        self.dim_y = self.args.dim_y
        self.layers = self.args.layers
        self.num_epochs = self.args.num_epochs
        self.lr = self.args.lr
        self.loss_fn = torch.nn.MSELoss()
        if self.args.device is None:
            self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MonomialNeuralNetwork(
            input_size = self.args.dim_x, 
            output_size = self.args.dim_y, 
            layers = [50, 50], 
            power = 2
            )
        self.model = self.model.to(self.args.device)