import torch
import numpy as np
# from attrdict import AttrDict
from monomial_neural_network import *

n = 100  # number of data points
d = 2  # input dimension
teacher_k = [5, 5]  # number of hidden neurons at each layer of the teacher
student_k = [5, 5]  # number of hidden neurons at each layer of the student

# create a random teacher model
def generate_teacher_model(d: int, k: int, weights=None):
    """
    Generates a teacher model for a neural network with specified input dimension, hidden layers, and weights.
    Args:
        d (int): The input dimension of the neural network.
        k (int): The number of hidden layers in the neural network.
        weights (list, optional): A list of weights for the neural network. If None, weights are generated based on
                                  Soltanolkotabi et al. 2022. Hidden weights are random normal with mean 0 and variance
                                  1/sqrt(d), and output weights are all 1.
    Returns:
        MonomialNeuralNetwork: An instance of MonomialNeuralNetwork with the specified configuration.
    """
    if weights is None:
        # if no weights are provided, generate weights based on
        # Soltanolkotabi et al. 2022
        weights = []

        # hidden weights are all random normal with mean 0 and variance
        # 1/sqrt(d)
        dim_in = d
        for dim_out in k:
            weight = torch.randn(dim_out, dim_in) / np.sqrt(dim_in)
            weights.append(weight)
            dim_in = dim_out

        # output weights are all 1
        v = torch.ones((1, dim_in)).float()
        weights.append(v)

    teacher_model = MonomialNeuralNetwork(
        input_size=d,
        output_size=1,
        hidden_layers=k,
        power=2,
        weights=weights
    )
    return teacher_model

# create a random teacher model with unit output weights
def generate_teacher_model_noOutWeight(d: int, k: int, weights=None):
    """
    Generates a teacher model for a neural network with specified input dimension, hidden layers, and weights.
    Args:
        d (int): The input dimension of the neural network.
        k (int): The number of hidden layers in the neural network.
        weights (list, optional): A list of weights for the neural network. If None, weights are generated based on
                                  Soltanolkotabi et al. 2022. Hidden weights are random normal with mean 0 and variance
                                  1/sqrt(d), and output weights are all 1.
    Returns:
        MonomialNeuralNetwork: An instance of MonomialNeuralNetwork with the specified configuration.
    """
    if weights is None:
        # if no weights are provided, generate weights based on
        # Soltanolkotabi et al. 2022
        weights = []

        # hidden weights are all random normal with mean 0 and variance
        # 1/sqrt(d)
        dim_in = d
        for dim_out in k:
            weight = torch.randn(dim_out, dim_in) / np.sqrt(dim_in)
            weights.append(weight)
            dim_in = dim_out

        # output weights are all 1
        v = torch.ones((1, dim_in)).float()
        weights.append(v)

    teacher_model = MonomialNeuralNetwork_noOutputWeight(
        input_size=d,
        output_size=1,
        hidden_layers=k,
        power=2,
        weights=weights
    )
    return teacher_model

# generate data
def generate_data(n: int, d: int, teacher_model):
    """
    Generates synthetic data using a given teacher model.

    Parameters:
    n (int): Number of data points to generate.
    d (int): Dimensionality of each data point.
    teacher_model (callable): A model that generates labels for the data points.

    Returns:
    tuple: A tuple containing:
        - x (torch.Tensor): A tensor of shape (n, d) containing the generated data points.
        - y (torch.Tensor): A tensor containing the labels generated by the teacher model.
    """
    x = torch.randn(n, d)
    with torch.no_grad():
        y = teacher_model(x)
    return x, y

# create a student model
def generate_student_model(d, k, weights=None):
    """
    Generates a student model for a neural network with specified input dimension, hidden layers, and weights.

    Args:
        d (int): The input dimension of the neural network.
        k (list): A list containing the number of neurons in each hidden layer of the neural network.
        weights (list, optional): A list of weights for the neural network. If None, weights are generated based on
                                  Soltanolkotabi et al. 2022. Hidden weights are random normal with mean 0 and variance
                                  1/sqrt(d), and output weights are uniformly random from {-1, 1}.

    Returns:
        MonomialNeuralNetwork: An instance of MonomialNeuralNetwork with the specified configuration.
    """
    if weights is None:
        # if no weights are provided, generate weights based on
        # Soltanolkotabi et al. 2022
        weights = []

        # hidden weights are all random normal with mean 0 and variance
        # 1/sqrt(d)
        dim_in = d
        for dim_out in k:
            # generates a tensor filled with random numbers drawn from a normal
            # distribution with a mean of 0 and a variance of 1
            weight = torch.randn(dim_out, dim_in) / np.sqrt(dim_in)
            weights.append(weight)
            dim_in = dim_out

            # output weights are uniformly random from {-1, 1}
            v = torch.randint(0, 2, (1, dim_in)).float() * 2 - 1
            weights.append(v)

    student_model = MonomialNeuralNetwork(
        input_size=d,
        output_size=1,
        hidden_layers=k,
        power=2,
        weights=weights
    )
    return student_model

# create a student model
def generate_student_model_noOutWeight(d, k, weights=None):
    """
    Generates a student model for a neural network with specified input dimension, hidden layers, and weights.

    Args:
        d (int): The input dimension of the neural network.
        k (list): A list containing the number of neurons in each hidden layer of the neural network.
        weights (list, optional): A list of weights for the neural network. If None, weights are generated based on
                                  Soltanolkotabi et al. 2022. Hidden weights are random normal with mean 0 and variance
                                  1/sqrt(d), and output weights are uniformly random from {-1, 1}.

    Returns:
        MonomialNeuralNetwork: An instance of MonomialNeuralNetwork with the specified configuration.
    """
    if weights is None:
        # if no weights are provided, generate weights based on
        # Soltanolkotabi et al. 2022
        weights = []

        # hidden weights are all random normal with mean 0 and variance
        # 1/sqrt(d)
        dim_in = d
        for dim_out in k:
            # generates a tensor filled with random numbers drawn from a normal
            # distribution with a mean of 0 and a variance of 1
            weight = torch.randn(dim_out, dim_in) / np.sqrt(dim_in)
            weights.append(weight)
            dim_in = dim_out

            # output weights are uniformly random from {-1, 1}
            v = torch.randint(0, 2, (1, dim_in)).float() * 2 - 1
            weights.append(v)

    student_model = MonomialNeuralNetwork_noOutputWeight(
        input_size=d,
        output_size=1,
        hidden_layers=k,
        power=2,
        weights=weights
    )
    return student_model


# TODO: write the experiment class.

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

# default_args = AttrDict(
#     {
#     'n': 100, # number of data points
#     'k': [5, 5], # number of hidden neurons at each layer
#     'dim_x': 10, # input dimension
#     'dim_y': 1,
#     'layers': [50, 50],
#     'data': 10,
#     'num_epochs': 1000,
#     'lr': 0.01
#     }
# )

# class Experiment:
#     def __init__(self, args = AttrDict()):
#         self.args = default_args + args
#         self.teacher_W = self.args.teacher_W
#         self.teacher_v = self.args.teacher_v
#         self.dim_x = self.args.dim_x
#         self.dim_y = self.args.dim_y
#         self.layers = self.args.layers
#         self.num_epochs = self.args.num_epochs
#         self.lr = self.args.lr
#         self.loss_fn = torch.nn.MSELoss()
#         if self.args.device is None:
#             self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = MonomialNeuralNetwork(
#             input_size = self.args.dim_x,
#             output_size = self.args.dim_y,
#             layers = [50, 50],
#             power = 2
#             )
#         self.model = self.model.to(self.args.device)

def pop_loss(student, teacher, d, N=1000):
    # compute the population loss, or at least estimate it by evaluating the function for a bunch of 
    # new data points
    # can be viewed as the generalization success of the model
    # N controls how many new data points are made

    # the new test data
    test_x = torch.randn(N, d)
    y_teach = teacher(test_x)
    y_stud = student(test_x)

    e_torch = torch.norm(y_teach - y_stud)
    e = (e_torch.detach().numpy())**2 / N # normalize by the numpy of points
    return e