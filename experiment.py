import torch
import numpy as np
from attrdict import AttrDict
from monomial_neural_network import *

n = 100 # number of data points
d = 10 # input dimension
teacher_k = [5, 5] # number of hidden neurons at each layer of the teacher
student_k = [5, 5] # number of hidden neurons at each layer of the student

# create a random teacher model
def generate_teacher_model(d, k):
    weights = []

    # hidden weights are all random normal with mean 0 and variance 1/sqrt(d)
    dim_in = d
    for dim_out in k:
        weight = torch.randn(dim_out, dim_in)/np.sqrt(dim_in)
        weights.append(weight)
        dim_in = dim_out

    # output weights are all 1
    v = torch.ones((dim_in, 1)).float() 
    weights.append(v)

    teacher_model = MonomialNeuralNetwork(
        input_size = d, 
        output_size = 1, 
        hidden_layers = k, 
        power = 2, 
        weights = weights
    )
    return teacher_model

# generate data
def generate_data(n, d, teacher_model):
    x = torch.randn(n, d)
    with torch.no_grad():
        y = teacher_model(x)
    return x, y

# create a student model
def generate_student_model(d, k):
    weights = []

    # hidden weights are all random normal with mean 0 and variance 1/sqrt(d)
    dim_in = d
    for dim_out in k:
        # generates a tensor filled with random numbers drawn from a normal 
        # distribution with a mean of 0 and a variance of 1
        weight = torch.randn(dim_out, dim_in)/np.sqrt(dim_in)
        weights.append(weight)
        dim_in = dim_out

    # output weights are uniformly random from {-1, 1} 
    v = torch.randint(0, 2, (dim_in, 1)).float() * 2 - 1 
    weights.append(v)
    student_model = MonomialNeuralNetwork(
        input_size = d, 
        output_size = 1, 
        hidden_layers = k, 
        power = 2, 
        weights = weights
    )
    return student_model

# train the student model

# evaluate the student model

# default_args = AttrDict(
#     {
#     'teacher_W': np.array([[1,2, 0.4],[1,-1, -0.2],[0,2, 0.9], [1,1, 0]]),
#     'teacher_v': np.array([[0, 1, 1, -0.2]]),
#     'dim_x': 2,
#     'dim_y': 1,
#     'layers': [50, 50],
#     'data': 10,
#     'num_epochs': 1000,
#     'lr': 0.01
#     }
# )

default_args = AttrDict(
    {
    'n': 100, # number of data points
    'k': [5, 5], # number of hidden neurons at each layer
    'dim_x': 10, # input dimension
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